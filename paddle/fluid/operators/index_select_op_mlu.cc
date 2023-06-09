/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {
using Tensor = phi::DenseTensor;

template <typename T>
class IndexSelectMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* index = ctx.Input<Tensor>("Index");
    auto dim = ctx.Attr<int>("dim");
    auto* out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc index_desc(*index);
    MLUCnnlTensorDesc out_desc(*out);
    MLUCnnl::IndexSelect(ctx,
                         dim,
                         x_desc.get(),
                         GetBasePtr(x),
                         index_desc.get(),
                         GetBasePtr(index),
                         out_desc.get(),
                         GetBasePtr(out));
  }
};

template <typename T>
class IndexSelectGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x_grad = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* index = ctx.Input<phi::DenseTensor>("Index");
    auto* out_grad = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));

    auto x_dims = x_grad->dims();
    auto out_dims = out_grad->dims();

    int dim = ctx.Attr<int>("dim");
    if (dim < 0) {
      dim += out_dims.size();
    }
    Tensor casted_index(index->dtype());
    MLUCnnlTensorDesc index_desc(*index);

    MLUCnnlTensorDesc out_grad_desc(*out_grad);
    if (index->dtype() != DataType::INT32) {
      casted_index.mutable_data<int32_t>(index->dims(), ctx.GetPlace());
      cnnlCastDataType_t cast_type =
          GetCastDataType(index->dtype(), DataType::INT32);
      MLUCnnlTensorDesc casted_index_desc(casted_index);
      MLUCnnl::Cast(ctx,
                    cast_type,
                    index_desc.get(),
                    GetBasePtr(index),
                    casted_index_desc.get(),
                    GetBasePtr(&casted_index));
    } else {
      casted_index = *index;
    }

    if (dim == 0) {
      MLUCnnlTensorDesc casted_index_desc(casted_index);
      x_grad->mutable_data<T>(ctx.GetPlace());
      MLUCnnlTensorDesc x_grad_desc(*x_grad);
      auto value_t = static_cast<T>(0.0f);
      MLUCnnl::Fill(ctx,
                    CNNL_POINTER_MODE_HOST,
                    &value_t,
                    x_grad_desc.get(),
                    GetBasePtr(x_grad));

      MLUCnnl::UnsortedSegmentSum(
          ctx,
          out_grad_desc.get(),
          GetBasePtr(out_grad),
          casted_index_desc.get(),
          static_cast<const int*>(GetBasePtr(&casted_index)),
          x_grad_desc.get(),
          GetBasePtr(x_grad));
    } else {
      Tensor transed_out_grad;
      std::vector<int> in_trans_perm;
      in_trans_perm.push_back(dim);
      for (int i = 0; i < out_dims.size(); ++i) {
        if (i == dim) continue;
        in_trans_perm.push_back(i);
      }
      framework::DDim transed_out_dims(out_dims);
      for (size_t i = 0; i < in_trans_perm.size(); ++i) {
        transed_out_dims[i] = out_dims[in_trans_perm[i]];
      }
      transed_out_grad.mutable_data<T>(transed_out_dims, ctx.GetPlace());
      MLUCnnlTensorDesc transed_out_grad_desc(transed_out_grad);
      const int in_trans_dim_size = in_trans_perm.size();
      MLUCnnl::Transpose(ctx,
                         in_trans_perm,
                         in_trans_dim_size,
                         out_grad_desc.get(),
                         GetBasePtr(out_grad),
                         transed_out_grad_desc.get(),
                         GetBasePtr(&transed_out_grad));

      Tensor sum_out;
      framework::DDim sum_dims(x_dims);
      sum_dims[0] = x_dims[dim];
      auto idx = 1;
      for (int i = 0; i < x_dims.size(); ++i) {
        if (i == dim) continue;
        sum_dims[idx++] = x_dims[i];
      }
      sum_out.mutable_data<T>(sum_dims, ctx.GetPlace());
      auto value_t = static_cast<T>(0.0f);
      MLUCnnlTensorDesc sum_out_desc(sum_out);
      MLUCnnl::Fill(ctx,
                    CNNL_POINTER_MODE_HOST,
                    &value_t,
                    sum_out_desc.get(),
                    GetBasePtr(&sum_out));
      MLUCnnlTensorDesc casted_index_desc(casted_index);

      MLUCnnl::UnsortedSegmentSum(
          ctx,
          transed_out_grad_desc.get(),
          GetBasePtr(&transed_out_grad),
          casted_index_desc.get(),
          static_cast<const int*>(GetBasePtr(&casted_index)),
          sum_out_desc.get(),
          GetBasePtr(&sum_out));

      std::vector<int> out_trans_perm;
      for (int i = 1; i < 1 + dim; ++i) {
        out_trans_perm.push_back(i);
      }
      out_trans_perm.push_back(0);
      for (int i = 1 + dim; i < x_dims.size(); ++i) {
        out_trans_perm.push_back(i);
      }
      x_grad->mutable_data<T>(ctx.GetPlace());
      MLUCnnlTensorDesc x_grad_desc(*x_grad);
      const int out_trans_dim_size = out_trans_perm.size();
      MLUCnnl::Transpose(ctx,
                         out_trans_perm,
                         out_trans_dim_size,
                         sum_out_desc.get(),
                         GetBasePtr(&sum_out),
                         x_grad_desc.get(),
                         GetBasePtr(x_grad));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_MLU_KERNEL(index_select,
                       ops::IndexSelectMLUKernel<float>,
                       ops::IndexSelectMLUKernel<plat::float16>,
                       ops::IndexSelectMLUKernel<int>,
                       ops::IndexSelectMLUKernel<int64_t>);

REGISTER_OP_MLU_KERNEL(index_select_grad,
                       ops::IndexSelectGradMLUKernel<float>,
                       ops::IndexSelectGradMLUKernel<plat::float16>,
                       ops::IndexSelectGradMLUKernel<int>,
                       ops::IndexSelectGradMLUKernel<int64_t>);
