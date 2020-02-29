from torch.nn.modules.module import Module
from torch.autograd import Function, Variable
import block_extractor_cuda

class BlockExtractorFunction(Function):

    @staticmethod
    def forward(ctx, source, flow_field, kernel_size):
        assert source.is_contiguous()
        assert flow_field.is_contiguous()

        # TODO: check the shape of the inputs 
        bs, ds, hs, ws = source.size()
        bf, df, hf, wf = flow_field.size()
        # assert bs==bf and hs==hf and ws==wf
        assert df==2

        ctx.save_for_backward(source, flow_field)
        ctx.kernel_size = kernel_size

        output = flow_field.new(bs, ds, kernel_size*hf, kernel_size*wf).zero_()

        if not source.is_cuda:
            raise NotImplementedError
        else:
            block_extractor_cuda.forward(source, flow_field, output, kernel_size)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output.contiguous()
        source, flow_field = ctx.saved_tensors
        grad_source = Variable(source.new(source.size()).zero_())
        grad_flow_field = Variable(flow_field.new(flow_field.size()).zero_())

        block_extractor_cuda.backward(source, flow_field, grad_output.data,
                                 grad_source.data, grad_flow_field.data, 
                                 ctx.kernel_size)

        return grad_source, grad_flow_field, None


class BlockExtractor(Module):
    def __init__(self, kernel_size=3):
        super(BlockExtractor, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, source, flow_field):
        source_c = source.contiguous()
        flow_field_c = flow_field.contiguous()
        return BlockExtractorFunction.apply(source_c, flow_field_c,  
                                          self.kernel_size)
