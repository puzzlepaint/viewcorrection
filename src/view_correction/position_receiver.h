// Copyright 2018 ETH Zürich
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef VIEW_CORRECTION_POSITION_RECEIVER_H_
#define VIEW_CORRECTION_POSITION_RECEIVER_H_

#include <mutex>
#include <thread>

#include <Eigen/Core>

namespace view_correction {

class PositionReceiver {
 public:
  PositionReceiver();
  ~PositionReceiver();
  
  bool Initialize(uint16_t port);
  
  // Call only one of the following.
  void StartReceiveThread();
  void ReceiveNonBlocking();
  
  inline bool received_any_observer_position() const {
    std::unique_lock<std::mutex> lock(access_mutex_);
    return received_any_observer_position_;
  }
  inline const Eigen::Vector3f last_received_observer_position() const {
    std::unique_lock<std::mutex> lock(access_mutex_);
    return last_received_observer_position_;
  }
  
 private:
  void ReceiveThreadMain();
  
  bool received_any_observer_position_;
  float last_received_pose_timestamp_;
  Eigen::Vector3f last_received_observer_position_;
  
  int udp_socket_fd_;
  
  mutable std::mutex access_mutex_;
  std::unique_ptr<std::thread> receive_thread_;
};

}  // namespace view_correction

#endif  // VIEW_CORRECTION_POSITION_RECEIVER_H_
