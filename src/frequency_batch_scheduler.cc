// Copyright 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "frequency_batch_scheduler.h"

#ifndef _WIN32
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include "constants.h"
#include "server.h"
#include "triton/common/logging.h"
#include "triton/common/model_config.h"
#include "triton/common/nvtx.h"

namespace triton { namespace core {

extern bool IsStaleState(Payload::State payload_state);
// bool
// IsStaleState(Payload::State payload_state)
// {
//   return (
//       (payload_state == Payload::State::EXECUTING) ||
//       (payload_state == Payload::State::RELEASED));
// }

Status
FrequencyQueue::Enqueue(std::unique_ptr<InferenceRequest>& request)
{
  queue_.push_back(std::move(request));
  return Status::Success;
}

Status
FrequencyQueue::Dequeue(std::unique_ptr<InferenceRequest>* request)
{
  if (queue_.empty()) {
    return Status(
        Status::Code::INTERNAL,
        "attempt to dequeue from empty frequency queue");
  }

  *request = std::move(queue_.front());
  queue_.pop_front();
  cursor_--;
  LOG_VERBOSE(1) << "FrequencyQueue dequeued request "
                 << " queue size = " << queue_.size() << " cursor = " << cursor_;
  return Status::Success;
}

size_t
FrequencyQueue::FindTimeout(const uint64_t oldest_timestamp)
{
  size_t batch_size = 0;
  for (size_t i = cursor_; i < queue_.size(); i++) {
    if (queue_[i]->QueueStartNs() < oldest_timestamp) {
      cursor_++;
      batch_size += queue_[i]->BatchSize();
    } else {
      break;
    }
  }
  LOG_VERBOSE(1) << "FrequencyQueue found timeout batch size = " << batch_size << " queue size = "
                 << queue_.size() << " cursor = " << cursor_;
  return batch_size;
}

size_t
FrequencyQueue::TryToBatch(const uint64_t optimal_batch_size)
{
  size_t batch_size = 0;
  uint64_t cursor = 0;
  bool send_batch = false;
  for (size_t i = 0; i < queue_.size(); i++) {
    batch_size += queue_[i]->BatchSize();
    if (batch_size > optimal_batch_size) {
      send_batch = true;
      break;
    }
    cursor++;
    if (batch_size == optimal_batch_size) {
      send_batch = true;
      break;
    }
  }
  if (send_batch) {
    cursor_ = std::max(cursor_, cursor);
    LOG_VERBOSE(1) << "FrequencyQueue found batch size = " << batch_size
                   << " optimal batch size = " << optimal_batch_size
                   << " queue size = " << queue_.size()
                   << " cursor = " << cursor_;
    return batch_size;
  }
  LOG_VERBOSE(1) << "FrequencyQueue failed to batch " << batch_size
                 << " optimal batch size = " << optimal_batch_size
                 << " queue size = " << queue_.size()
                 << " cursor = " << cursor_;
  return 0;
}

FrequencyBatchScheduler::FrequencyBatchScheduler(
    TritonModel* model, TritonModelInstance* model_instance,
    const bool frequency_batching_enabled, const int32_t max_batch_size,
    const int32_t optimal_batch_size,
    const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
    const bool preserve_ordering, const bool response_cache_enable,
    const uint64_t max_queue_delay_microseconds)
    : model_(model),
      model_instance_(model_instance),
      frequency_batching_enabled_(frequency_batching_enabled),
      queue_(),
      stop_(false),
      max_batch_size_((size_t)std::max(1, max_batch_size)),
      optimal_batch_size_((size_t)std::max(1, optimal_batch_size)),
      next_preferred_batch_size_(0),
      pending_batch_delay_ns_(max_queue_delay_microseconds * 1000),
      queued_batch_size_(0),
      enforce_equal_shape_tensors_(enforce_equal_shape_tensors),
      has_optional_input_(false),
      preserve_ordering_(preserve_ordering) {
  LOG_VERBOSE(1) << "Creating FrequencyBatchScheduler for model '"
                 << model_->Name() << "' version '" << model_->Version() << "'"
                 << " max_batch_size " << max_batch_size_
                 << " optimal_batch_size " << optimal_batch_size_;
  rate_limiter_ = model_->Server()->GetRateLimiter();
  // Both the server and model config should specify
  // caching enabled for model to utilize response cache.
  response_cache_enabled_ =
      (model_->Server()->ResponseCacheEnabled() && response_cache_enable);
#ifdef TRITON_ENABLE_METRICS
  // Initialize metric reporter for cache statistics if cache enabled
  if (response_cache_enabled_) {
    MetricModelReporter::Create(
        model_->Name(), model_->Version(), METRIC_REPORTER_ID_RESPONSE_CACHE,
        model_->Config().metric_tags(), &reporter_);
  }
#endif  // TRITON_ENABLE_METRICS

  for (const auto& input : model_->Config().input()) {
    if (input.optional()) {
      has_optional_input_ = true;
      break;
    }
  }
}

Status
FrequencyBatchScheduler::Create(
    TritonModel* model, TritonModelInstance* model_instance, const int nice,
    const bool frequency_batching_enabled, const int32_t max_batch_size,
    const int32_t optimal_batch_size,
    const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
    const bool preserve_ordering, const bool response_cache_enable,
    const uint64_t max_queue_delay_microseconds,
    std::unique_ptr<Scheduler>* scheduler)
{
  inference::ModelFrequencyBatching batcher_config;
  batcher_config.set_preserve_ordering(preserve_ordering);
  batcher_config.set_max_queue_delay_microseconds(max_queue_delay_microseconds);

  return Create(
      model, model_instance, nice, frequency_batching_enabled, max_batch_size,
      enforce_equal_shape_tensors, batcher_config,
      response_cache_enable, scheduler);
}

Status
FrequencyBatchScheduler::Create(
    TritonModel* model, TritonModelInstance* model_instance, const int nice,
    const bool frequency_batching_enabled, const int32_t max_batch_size,
    const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
    const inference::ModelFrequencyBatching& batcher_config,
    const bool response_cache_enable, std::unique_ptr<Scheduler>* scheduler)
{
  FrequencyBatchScheduler* dyna_sched = new FrequencyBatchScheduler(
      model, model_instance, frequency_batching_enabled, max_batch_size,
      batcher_config.optimal_batch_size(), enforce_equal_shape_tensors,
      batcher_config.preserve_ordering(), response_cache_enable,
      batcher_config.max_queue_delay_microseconds());
  std::unique_ptr<FrequencyBatchScheduler> sched(dyna_sched);

  sched->scheduler_thread_exit_.store(false);
  if (frequency_batching_enabled) {
    sched->scheduler_thread_ =
        std::thread([dyna_sched, nice]() { dyna_sched->BatcherThread(nice); });
  }

  scheduler->reset(sched.release());

  return Status::Success;
}

FrequencyBatchScheduler::~FrequencyBatchScheduler()
{
  // Signal the scheduler thread to exit and then wait for it..
  scheduler_thread_exit_.store(true);
  cv_.notify_one();
  if (scheduler_thread_.joinable()) {
    scheduler_thread_.join();
  }
}

Status
FrequencyBatchScheduler::Enqueue(std::unique_ptr<InferenceRequest>& request)
{
  if (stop_) {
    return Status(
        Status::Code::UNAVAILABLE,
        request->LogRequest() +
            "Server is stopping, scheduler for model has stopped accepting new "
            "inference requests");
  }
  // If queue start timestamp hasn't been set, queue timer starts at
  // the beginning of the queueing and scheduling process. Otherwise,
  // frequency batcher is used as component of another batcher and should not
  // overwrite the queue start timestamp.
  if (request->QueueStartNs() == 0) {
    request->CaptureQueueStartNs();
    INFER_TRACE_ACTIVITY(
        request->Trace(), TRITONSERVER_TRACE_QUEUE_START,
        request->QueueStartNs());
#ifdef TRITON_ENABLE_TRACING
    request->TraceInputTensors(
        TRITONSERVER_TRACE_TENSOR_QUEUE_INPUT, "FrequencyBatchScheduler Enqueue");
#endif  // TRITON_ENABLE_TRACING
  }

  // Record time at the beginning of the batcher queueing. In the case of
  // oldest sequence batcher, this will overwrite the value that was previously
  // set by sequence batcher, which is okay as by this point, the previous
  // batcher won't be needing this value and it can be safely reused by
  // the frequency batcher.
  request->CaptureBatcherStartNs();

  std::unique_ptr<InferenceResponse> cached_response;

  if (response_cache_enabled_) {
    CacheLookUp(request, cached_response);
  }

  if (cached_response != nullptr) {
    // If there was a cache hit then try sending the cached response
    // and release the request.
    if (preserve_ordering_) {
      // In order to preserve the order, the response send must be
      // delegated.
      DelegateResponse(request);
    }

    // Send cached response and release request
    InferenceResponse::Send(
        std::move(cached_response), TRITONSERVER_RESPONSE_COMPLETE_FINAL);
    InferenceRequest::Release(
        std::move(request), TRITONSERVER_REQUEST_RELEASE_ALL);

    return Status::Success;
  }

  if (!frequency_batching_enabled_) {
    if (preserve_ordering_ || response_cache_enabled_) {
      DelegateResponse(request);
    }
    // If not using frequency batching, directly enqueue the
    // request to model for execution
    auto payload = model_->Server()->GetRateLimiter()->GetPayload(
        Payload::Operation::INFER_RUN, nullptr /* TritonModelInstance*/);
    payload->AddRequest(std::move(request));
    RETURN_IF_ERROR(
        model_->Server()->GetRateLimiter()->EnqueuePayload(model_, payload));

  } else {
    bool wake_batcher = true;
    {
      std::lock_guard<std::mutex> lock(mu_);

      queued_batch_size_ += std::max(1U, request->BatchSize());

      // Assuming no error is returned, this call takes ownership of
      // 'request' and so we can't use it after this point.
      RETURN_IF_ERROR(queue_.Enqueue(request));

      LOG_VERBOSE(1) << "FrequencyBatchScheduler queue size: "
                     << queue_.Size() << ", queued_batch_size_: " << queued_batch_size_;

      // If there are any idle runners and the queued batch size is greater or
      // equal to next preferred batch size, then wake batcher up to service
      // this request. We do the actual wake outside of the lock to avoid
      // having the woken thread immediately block on the lock
      wake_batcher =
          model_->Server()->GetRateLimiter()->PayloadSlotAvailable(model_);

      // We may wake up runner less often if we don't enforce equal shape
      // within a batch, otherwise must always wake up runner to check it
      if (enforce_equal_shape_tensors_.empty()) {
        std::lock_guard<std::mutex> exec_lock(*(curr_payload_->GetExecMutex()));
        auto payload_state = curr_payload_->GetState();
        wake_batcher &=
            (payload_saturated_ || IsStaleState(payload_state) ||
             (queued_batch_size_ >= next_preferred_batch_size_));
      }
    }

    if (wake_batcher) {
      // log the reason for waking up the batcher
      if (enforce_equal_shape_tensors_.empty()) {
        std::lock_guard<std::mutex> exec_lock(*(curr_payload_->GetExecMutex()));
        auto payload_state = curr_payload_->GetState();
        if (payload_saturated_) {
          LOG_VERBOSE(1) << "FrequencyBatchScheduler wake up batcher due to "
                            "payload saturated";
        } else if (IsStaleState(payload_state)) {
          LOG_VERBOSE(1) << "FrequencyBatchScheduler wake up batcher due to "
                            "stale state";
        } else if (queued_batch_size_ >= optimal_batch_size_) {
          LOG_VERBOSE(1) << "FrequencyBatchScheduler wake up batcher due to "
                            "queued_batch_size_ >= optimal_batch_size_";
        }
      } else {
        LOG_VERBOSE(1) << "FrequencyBatchScheduler wake up batcher due to "
                          "enforce_equal_shape_tensors_";
      }
      cv_.notify_one();
    }
  }

  return Status::Success;
}

void
FrequencyBatchScheduler::NewPayload()
{
  curr_payload_ = model_->Server()->GetRateLimiter()->GetPayload(
      Payload::Operation::INFER_RUN, model_instance_);
  payload_saturated_ = false;
}

void
FrequencyBatchScheduler::BatcherThread(const int nice)
{
#ifndef _WIN32
  if (setpriority(PRIO_PROCESS, syscall(SYS_gettid), nice) == 0) {
    LOG_VERBOSE(1) << "Starting frequency-batcher thread for " << model_->Name()
                   << " at nice " << nice << "...";
  } else {
    LOG_VERBOSE(1) << "Starting frequency-batcher thread for " << model_->Name()
                   << " at default nice (requested nice " << nice
                   << " failed)...";
  }
#else
  LOG_VERBOSE(1) << "Starting frequency-batcher thread for " << model_->Name()
                 << " at default nice...";
#endif
  // For debugging/testing, delay start of threads until the queue
  // contains the specified number of entries.
  size_t delay_cnt = 0;
  {
    const char* dstr = getenv("TRITONSERVER_DELAY_SCHEDULER");
    if (dstr != nullptr) {
      delay_cnt = atoi(dstr);
      LOG_VERBOSE(1) << "Delaying batcher thread for " << model_->Name()
                     << " until " << delay_cnt << " queued requests...";
    }
  }

  auto wait_for_slots = [this]() {
    return model_->Server()->GetRateLimiter()->PayloadSlotAvailable(model_);
  };
  NewPayload();
  const uint64_t default_wait_microseconds = 500 * 1000;

  while (!scheduler_thread_exit_.load()) {
    NVTX_RANGE(nvtx_, "FrequencyBatcher " + model_->Name());

    std::shared_ptr<std::vector<std::deque<std::unique_ptr<InferenceRequest>>>>
        rejected_requests;
    uint64_t wait_microseconds = 0;

    // Hold the lock for as short a time as possible.
    {
      std::unique_lock<std::mutex> lock(mu_);
      {
        std::lock_guard<std::mutex> exec_lock(*(curr_payload_->GetExecMutex()));
        auto payload_state = curr_payload_->GetState();
        if (payload_saturated_ || IsStaleState(payload_state)) {
          NewPayload();
          required_equal_inputs_.clear();
        }
      }

      if (delay_cnt > 0) {
        // Debugging/testing... wait until queue contains 'delay_cnt'
        // items...
        wait_microseconds = 10 * 1000;
        if (queue_.Size() >= delay_cnt) {
          delay_cnt = 0;
        }
        LOG_VERBOSE(1) << "Delaying batcher thread " << model_->Name()
                       << " until " << delay_cnt
                       << " queued requests, current total = " << queue_.Size();
      } else if (queue_.Empty()) {
        wait_microseconds = default_wait_microseconds;
      } else {
        if (payload_saturated_) {
          continue;
        }
        cv_.wait(lock, wait_for_slots);
        {
          std::lock_guard<std::mutex> exec_lock(
              *(curr_payload_->GetExecMutex()));

          auto payload_state = curr_payload_->GetState();
          if (IsStaleState(payload_state)) {
            continue;
          }

          // // Use frequency batching to get request(s) to execute.
          // wait_microseconds = GetFrequencyBatch();

          wait_microseconds = GetFrequencyBatch();
          size_t pending_batch_queue_cnt = queue_.Cursor();
          
          LOG_VERBOSE(1) << "Frequency thread for " << model_->Name()
                << " pending batch queue count " << pending_batch_queue_cnt
                << " wait microseconds " << wait_microseconds;

          if (wait_microseconds == 0 && pending_batch_queue_cnt > 0) {
            size_t sent_batch_size = 0;
            curr_payload_->ReserveRequests(pending_batch_queue_cnt);
            for (size_t idx = 0; idx < pending_batch_queue_cnt; ++idx) {
              std::unique_ptr<InferenceRequest> request;
              auto status = queue_.Dequeue(&request);
              if (status.IsOk()) {
                if (preserve_ordering_ || response_cache_enabled_) {
                  DelegateResponse(request);
                }
                sent_batch_size += request->BatchSize();
                curr_payload_->AddRequest(std::move(request));
              } else {
                // The queue is empty which conflicts with pending batch
                // count. Send the current batch if any and reset related
                // variables.
                LOG_ERROR << request->LogRequest()
                          << "Failed to retrieve request from scheduler queue: "
                          << status.Message();
                queued_batch_size_ = 0;
                break;
              }
            }

            if (curr_payload_->GetState() == Payload::State::UNINITIALIZED) {
              curr_payload_->SetState(Payload::State::READY);
            }
            queued_batch_size_ -= sent_batch_size;
            LOG_VERBOSE(1) << "Frequency batcher thread " << model_->Name()
                           << " sent " << pending_batch_queue_cnt
                           << " requests, queue size = " << queue_.Size()
                           << " queued batch size = " << queued_batch_size_;
          }
        }
      }

      // If no requests are to be handled, wait for notification or
      // for the specified timeout before checking the queue again.
      if (wait_microseconds > 0) {
        std::chrono::microseconds wait_timeout(wait_microseconds);
        cv_.wait_for(lock, wait_timeout);
      }
    }

    if (curr_payload_->GetState() == Payload::State::READY) {
      auto callback = [this]() { cv_.notify_one(); };
      curr_payload_->SetCallback(callback);
      model_->Server()->GetRateLimiter()->EnqueuePayload(model_, curr_payload_);
    }

    // Finish rejected requests if any
    if (rejected_requests != nullptr) {
      static Status rejected_status =
          Status(Status::Code::UNAVAILABLE, "Request timeout expired");
      for (auto& rejected_queue : *rejected_requests) {
        for (auto& rejected_request : rejected_queue) {
          InferenceRequest::RespondIfError(
              rejected_request, rejected_status, true);
        }
      }
    }
  }  // end runner loop

  LOG_VERBOSE(1) << "Stopping frequency-batcher thread for " << model_->Name()
                 << "...";
}

uint64_t
FrequencyBatchScheduler::GetFrequencyBatch()
{
  // 'mu_' mutex must be held when this function is called. queue_
  // must not be empty.

  // Examine the new requests. If adding these new requests to the
  // pending batch allows a preferred batch size then execute it
  // immediately. Stop examining requests if the maximum preferred
  // batch size would be exceeded or if the shape of the next request
  // does not match the shape of the pending batch.
  // bool send_now = false;

  auto payload_batch_size = curr_payload_->BatchSize();

  // get time now ns
  uint64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
  size_t timeout_batch_size = queue_.FindTimeout(now_ns - pending_batch_delay_ns_);
  if (payload_batch_size + timeout_batch_size >= optimal_batch_size_) {
    return 0;  // send these requests now
  }

  size_t remaining_batch_size = optimal_batch_size_ - payload_batch_size;
  size_t actual_batch_size = queue_.TryToBatch(remaining_batch_size);
  LOG_VERBOSE(1) << "Frequency batcher thread " << model_->Name()
                 << " timeout_batch_size = " << timeout_batch_size
                 << " payload_batch_size = " << payload_batch_size
                  << " actual_batch_size = " << actual_batch_size;
  if (timeout_batch_size > 0) {
    next_preferred_batch_size_ = 0;
  } else if (actual_batch_size > 0) {
    next_preferred_batch_size_ = actual_batch_size;
  }
  if (timeout_batch_size + actual_batch_size > 0) {
    return 0;
  }
  
  return 5 * 1000; // wait 5 ms
}

void
FrequencyBatchScheduler::DelegateResponse(
    std::unique_ptr<InferenceRequest>& request)
{
  std::lock_guard<std::mutex> lock(completion_queue_mtx_);
  completion_queue_.emplace_back();
  auto queue_slot = &completion_queue_.back();
  // Pass raw ptr to lambda for tracking stats from cache and updating
  // metric reporter on cache miss stats after insertion
  InferenceRequest* raw_request_ptr = request.get();

  request->SetResponseDelegator(
      [this, queue_slot, raw_request_ptr](
          std::unique_ptr<InferenceResponse>&& response, const uint32_t flags) {
        if (response_cache_enabled_ && raw_request_ptr->CacheKeyIsSet()) {
          // Cache insertion happens here because we need the backend to have
          // computed the inference response first in the case of cache miss
          auto cache = model_->Server()->GetResponseCache();
          auto status = cache->Insert(*response, raw_request_ptr);
          bool cache_miss =
              (status.StatusCode() != Status::Code::ALREADY_EXISTS);
          if (cache_miss) {
#ifdef TRITON_ENABLE_STATS
            // Update cache miss statistics even on failure to insert
            // as we still spend time on lookup and attempting to insert
            raw_request_ptr->ReportStatisticsCacheMiss(reporter_.get());
#endif  // TRITON_ENABLE_STATS

            if (!status.IsOk()) {
              LOG_ERROR << raw_request_ptr->LogRequest()
                        << "Failed to insert request_hash ["
                        << raw_request_ptr->CacheKey()
                        << "] into response cache: " << status.Message();
            }
          }  // Otherwise do nothing; we update cache hit statistics on Lookup
        }

        if (preserve_ordering_) {
          {
            std::lock_guard<std::mutex> lock(completion_queue_mtx_);
            queue_slot->emplace_back(std::move(response), flags);
          }
          FinalizeResponses();
        } else {
          InferenceResponse::Send(std::move(response), flags);
        }
      });
}

void
FrequencyBatchScheduler::CacheLookUp(
    std::unique_ptr<InferenceRequest>& request,
    std::unique_ptr<InferenceResponse>& cached_response)
{
  auto cache = model_->Server()->GetResponseCache();
  // Lookup request in cache
  std::unique_ptr<InferenceResponse> local_response;
  request->ResponseFactory().CreateResponse(&local_response);
  auto status = cache->Lookup(local_response.get(), request.get());
  if (status.IsOk() && (local_response != nullptr)) {
    cached_response = std::move(local_response);
#ifdef TRITON_ENABLE_STATS
    // Update model metrics/stats on cache hits
    // Backends will update metrics as normal on cache misses
    request->ReportStatisticsCacheHit(reporter_.get());
#endif  // TRITON_ENABLE_STATS
  }
}

void
FrequencyBatchScheduler::FinalizeResponses()
{
  // Need exclusive access of the function to ensure responses are sent
  // in order
  static std::mutex finalize_mtx;
  std::lock_guard<std::mutex> lock(finalize_mtx);
  // Finalize the completed payloads in-order as far as possible
  std::deque<std::pair<std::unique_ptr<InferenceResponse>, const uint32_t>>
      responses;
  {
    std::lock_guard<std::mutex> queue_lock(completion_queue_mtx_);
    while (!completion_queue_.empty() && !completion_queue_.front().empty()) {
      bool response_complete = false;
      for (auto& response_pair : completion_queue_.front()) {
        // Assuming FINAL flag is set only in the last response of the request
        response_complete =
            ((response_pair.second & TRITONSERVER_RESPONSE_COMPLETE_FINAL) !=
             0);
        responses.emplace_back(std::move(response_pair));
      }
      if (response_complete) {
        completion_queue_.pop_front();
      } else {
        completion_queue_.front().clear();
      }
    }
  }

  for (auto& response : responses) {
    InferenceResponse::Send(std::move(response.first), response.second);
  }
}
}}  // namespace triton::core
