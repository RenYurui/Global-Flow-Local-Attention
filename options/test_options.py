from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self,  parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of the test examples')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here')
        # parser.add_argument('--phase', type=str, default='test', help='train, val, test')
        # parser.add_argument('--nsampling', type=int, default=1, help='ramplimg # times for each images')
        # parser.add_argument('--save_number', type=int, default=10, help='choice # reasonable results based on the discriminator score')

        parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=256, load_size=256, display_winsize=256)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        parser.set_defaults(batchSize=1)
        self.isTrain = False

        return parser
