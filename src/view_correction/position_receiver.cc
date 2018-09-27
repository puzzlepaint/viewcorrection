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

#include "view_correction/position_receiver.h"

#include <arpa/inet.h>
#include <glog/logging.h>
#include <limits>
#include <netinet/in.h>
#include <string.h>
#include <sys/socket.h>

namespace view_correction {

// Unfortunately, this struct is private in the UDPServer, so redefine it here.
struct UDPPosePacket {
  float timestamp, position_x_meters, position_y_meters, position_z_meters,
      quaternion_x, quaternion_y, quaternion_z, quaternion_w;
};

PositionReceiver::PositionReceiver()
    : received_any_observer_position_(false),
      last_received_pose_timestamp_(-std::numeric_limits<float>::infinity()),
      udp_socket_fd_(-1) {}

PositionReceiver::~PositionReceiver() {
  if (udp_socket_fd_ >= 0) {
    close(udp_socket_fd_);
  }
}

bool PositionReceiver::Initialize(uint16_t port) {
  // Create a socket.
  if ((udp_socket_fd_ = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
    LOG(ERROR) << "Cannot create UDP socket.";
    return false;
  }
  
  // Bind the socket to an IP address and port.
  struct sockaddr_in local_endpoint;
  memset(&local_endpoint, 0, sizeof(local_endpoint));
  local_endpoint.sin_family = AF_INET;
  local_endpoint.sin_addr.s_addr = htonl(INADDR_ANY);
  local_endpoint.sin_port = htons(port);
  if (bind(udp_socket_fd_, (struct sockaddr *)&local_endpoint,
           sizeof(local_endpoint)) < 0) {
    LOG(ERROR) << "Bind failed.";
    return false;
  }
  return true;
}

void PositionReceiver::StartReceiveThread() {
  receive_thread_.reset(new std::thread(
      std::bind(&PositionReceiver::ReceiveThreadMain, this)));
}

void PositionReceiver::ReceiveNonBlocking() {
  constexpr int kBufferSize = 1024;
  uint8_t buffer[kBufferSize];
  while (true) {
    int received_bytes = recv(udp_socket_fd_, buffer, kBufferSize, MSG_DONTWAIT);
    if (received_bytes <= 0) {
      break;
    }
    
    if (received_bytes == sizeof(UDPPosePacket)) {
      UDPPosePacket* packet = reinterpret_cast<UDPPosePacket*>(buffer);
      
      LOG(INFO) << "Received pose message with timestamp: " << packet->timestamp;
      if (packet->timestamp <= last_received_pose_timestamp_) {
        // Packets may arrive out of order with UDP. Skip old packets.
        LOG(INFO) << "Received out-of-order pose message.";
        continue;
      }
      
      std::unique_lock<std::mutex> lock(access_mutex_);
      received_any_observer_position_= true;
      last_received_pose_timestamp_ = packet->timestamp;
      last_received_observer_position_ =
          Eigen::Vector3f(packet->position_x_meters, packet->position_y_meters,
                          packet->position_z_meters);
    } else {
      LOG(WARNING) << "Received message on pose port with wrong size.";
    }
  }
}

void PositionReceiver::ReceiveThreadMain() {
  constexpr int kBufferSize = 1024;
  uint8_t buffer[kBufferSize];
  while (true) {
    int received_bytes = recv(udp_socket_fd_, buffer, kBufferSize, 0);
    if (received_bytes <= 0) {
      continue;
    }
    
    if (received_bytes == sizeof(UDPPosePacket)) {
      UDPPosePacket* packet = reinterpret_cast<UDPPosePacket*>(buffer);
      
      LOG(INFO) << "Received pose message with timestamp: " << packet->timestamp;
      if (packet->timestamp <= last_received_pose_timestamp_) {
        // Packets may arrive out of order with UDP. Skip old packets.
        LOG(INFO) << "Received out-of-order pose message.";
        continue;
      }
      
      std::unique_lock<std::mutex> lock(access_mutex_);
      received_any_observer_position_= true;
      last_received_pose_timestamp_ = packet->timestamp;
      last_received_observer_position_ =
          Eigen::Vector3f(packet->position_x_meters, packet->position_y_meters,
                          packet->position_z_meters);
    } else {
      LOG(WARNING) << "Received message on pose port with wrong size.";
    }
  }
}

}  // namespace view_correction
