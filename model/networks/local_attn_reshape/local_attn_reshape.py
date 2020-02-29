from torch.nn.modules.module import Module
from torch.autograd import Function, Variable
import local_attn_reshape_cuda

class LocalAttnReshapeFunction(Function):

    @staticmethod
    def forward(ctx, inputs, kernel_size):
        assert inputs.is_contiguous()

        # TODO: check the shape of the inputs 
        bs, ds, hs, ws = inputs.size()
        assert ds == kernel_size*kernel_size

        ctx.save_for_backward(inputs)
        ctx.kernel_size = kernel_size

        output = inputs.new(bs, 1, kernel_size*hs, kernel_size*ws).zero_()

        if not inputs.is_cuda:
            raise NotImplementedError
        else:
            local_attn_reshape_cuda.forward(inputs, output, kernel_size)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output.contiguous()
        inputs, = ctx.saved_tensors
        grad_inputs = Variable(inputs.new(inputs.size()).zero_())

        local_attn_reshape_cuda.backward(inputs, grad_output.data,
                                 grad_inputs.data, ctx.kernel_size)

        return grad_inputs, None


class LocalAttnReshape(Module):
    def __init__(self):
        super(LocalAttnReshape, self).__init__()

    def forward(self, inputs, kernel_size=3):
        inputs_c = inputs.contiguous()
        return LocalAttnReshapeFunction.apply(inputs_c, kernel_size)
