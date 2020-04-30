from .base_options import BaseOptions


class ValOptions(BaseOptions):
    def initialize(self,  parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./demo_results/', help='saves results here')
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(phase='val')
        parser.set_defaults(batchSize=1)
        self.isTrain = False

        return parser
