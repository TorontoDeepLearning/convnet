#include "multigpu_convnet.h"

MultiGPUConvNet::MultiGPUConvNet(const string& model_file):
  ConvNet(model_file) {}

void MultiGPUConvNet::Fprop(bool train) {
  int dst_layer_gpu_id, edge_gpu_id, src_layer_gpu_id;
  Layer *src_layer;
  Matrix *dst, *src;

  const int num_gpus = Matrix::GetNumBoards();
  vector<bool> overwrite(num_gpus);
  for(Layer* l : layers_) {
    for (int i = 0; i < num_gpus; i++) overwrite[i] = true;
    dst_layer_gpu_id = l->GetGPUId();
    for (Edge* e : l->incoming_edge_) {
      src_layer = e->GetSource();
      edge_gpu_id = e->GetGPUId();
      src_layer_gpu_id = src_layer->GetGPUId();

      if (edge_gpu_id != dst_layer_gpu_id) {
        dst = &(l->GetOtherState(edge_gpu_id));
      } else {
        dst = &(l->GetState());
      }
      if (edge_gpu_id != src_layer_gpu_id) {
        src = &(src_layer->GetOtherState(edge_gpu_id));
      } else {
        src = &(src_layer->GetState());
      }

      e->ComputeUp(*src, *dst, overwrite[edge_gpu_id]);
      overwrite[edge_gpu_id] = false;
    }
    
    if (l->IsInput()) {
      Matrix::SetDevice(l->GetGPUId());
      l->ApplyDropout(train);
      l->GetState().SetReady();
      l->SyncOutgoingState();
    } else {
      l->SyncIncomingState();
      l->ApplyActivation(train);
      l->SyncOutgoingState();
    }
  }
}

void MultiGPUConvNet::Bprop(bool update_weights) {
  Layer *l, *dst_layer;
  const int num_gpus = Matrix::GetNumBoards();
  vector<bool> overwrite(num_gpus);
  int src_layer_gpu_id, edge_gpu_id, dst_layer_gpu_id;
  Matrix *src_deriv, *dst_deriv, *src_state, *dst_state;
  for (int i = layers_.size() - 1; i >= 0; i--) {
    for (int i = 0; i < num_gpus; i++) overwrite[i] = true;
    l = layers_[i];
    src_layer_gpu_id = l->GetGPUId();
    for (Edge* e : l->outgoing_edge_) {
      dst_layer = e->GetDest();
      edge_gpu_id = e->GetGPUId();
      dst_layer_gpu_id = dst_layer->GetGPUId();

      if (edge_gpu_id != src_layer_gpu_id) {
        src_deriv = &(l->GetOtherDeriv(edge_gpu_id));
        src_state = &(l->GetOtherState(edge_gpu_id));
      } else {
        src_deriv = &(l->GetDeriv());
        src_state = &(l->GetState());
      }
      if (edge_gpu_id != dst_layer_gpu_id) {
        dst_deriv = &(dst_layer->GetOtherDeriv(edge_gpu_id));
        dst_state = &(dst_layer->GetOtherState(edge_gpu_id));
      } else {
        dst_deriv = &(dst_layer->GetDeriv());
        dst_state = &(dst_layer->GetState());
      }

      e->ComputeOuter(*src_state, *dst_deriv);
      if (!l->IsInput() && !e->IsBackPropBlocked()) {
        e->ComputeDown(*dst_deriv, *src_state, *dst_state, *src_deriv,
                       overwrite[edge_gpu_id]);
      }
      if (update_weights) e->UpdateWeights();
      overwrite[edge_gpu_id] = false;
    }
    if (!l->IsInput()) {
      if (!l->IsOutput()) {
        l->SyncOutgoingDeriv();
        l->ApplyDerivativeOfActivation();
      }
      l->SyncIncomingDeriv();
    }
  }
}
