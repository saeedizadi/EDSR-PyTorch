def set_template(args):
    # Set the templates here
    if args.template.find('jpeg') >= 0:
        args.data_train = 'DIV2K_jpeg'
        args.data_test = 'DIV2K_jpeg'
        args.epochs = 200
        args.decay = '100'

    if args.template.find('CLE') >= 0:
        args.data_train = 'CLE'
        args.data_test = 'CLE'
        args.dir_data = '../../data'
        args.n_colors = 1

    if args.template.find('CLE800') >= 0:
        args.data_train = 'CLE'
        args.data_test = 'CLE'
        args.dir_data = '../../data/data3'
        args.n_colors = 1

    if args.template.find('CLE200') >= 0:
        args.data_train = 'CLE'
        args.data_test = 'CLE'
        args.dir_data = '../../data/data2'
        args.n_colors = 1

    if args.template.find('EDSR_paper') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

    if args.template.find('DDBPN') >= 0:
        args.model = 'DDBPN'
        args.patch_size = 128
        args.scale = '4'

        args.data_test = 'Set5'

        args.batch_size = 20
        args.epochs = 1000
        args.decay = '500'
        args.gamma = 0.1
        args.weight_decay = 1e-4

        args.loss = '1*MSE'

    if args.template.find('GAN') >= 0:
        args.epochs = 200
        args.lr = 5e-5
        args.decay = '150'

    if args.template.find('RCAN') >= 0:
        args.model = 'RCAN'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64
        args.chop = True

    if args.template.find('VDSR') >= 0:
        args.model = 'VDSR'
        args.n_resblocks = 20
        args.n_feats = 64
        args.patch_size = 41
        args.lr = 1e-4
        args.batch_size = 64
        # args.momentum = 0.9
        # args.weight_decay = 0.0001
        # args.optimizer = 'SGD'
        args.epochs = 80
        # args.decay = '20-40-60'
        # args.gamma = 0.1

    if args.template.find('SRCNN') >= 0:
        args.model = 'SRCNN'
        args.n_feats = 64
        args.patch_size = 32
        args.lr = 1e-4
        # args.loss = '1*MSE'
        # args.optimizer = 'SGD'
        # args.momentum = 0.9
        # args.weight_decay = 0.0001
        # args.batch_size = 64

    if args.template.find('MEMNET') >= 0:
        args.model = 'MEMNET'
        args.n_feats = 64
        args.resgroups = 6
        args.resblocks = 6
        args.patch_size = 32
        args.optimizer = 'SGD'
        args.momentum = 0.9
        args.weight_decay = 0.0001
        args.batch_size = 16
        args.lr = 0.1
        args.epochs = 80
        args.decay = '20-40-60'
        args.gamma = 0.1

    if args.template.find('CARN') >= 0:
        args.model = 'CARN'
        args.patch_size = 48 * args.scale[0]
        args.batch_size = 64

    if args.template.find('DRCN') >= 0:
        args.model = 'DRCN'
        args.n_feats = 64
        args.patch_size = 41
        # args.optimizer = 'SGD'
        # args.momentum = 0.9
        # args.weight_decay = 0.0001
        args.batch_size = 32
        # args.lr = 0.01
        args.epochs = 80
        # args.decay = '20-40-60'
        # args.gamma = 0.1

    if args.template.find('DRRN') >= 0:
        args.model = 'DRRN'
        args.patch_size = 31
        args.optimizer = 'SGD'
        args.momentum = 0.9
        args.weight_decay = 0.0001
        args.batch_size = 128
        args.lr = 0.1
        args.epochs = 80
        args.decay = '10-20-30-40-50-60-70'

    if args.template.find('PHAM') >= 0:
        args.model = 'PHAM'
        args.n_resblocks = 5
        args.n_feats = 64
        args.patch_size = 48 * args.scale[0]
        args.epochs = 300

    if args.template.find('LapSRN') >= 0:
        args.model = 'LapSRN'
        args.patch_size = 128
        # args.optimizer = 'SGD'
        # args.momentum = 0.9
        # args.batch_size = 64
        # args.weight_decay = 0.0001
        args.batch_size = 64
        # args.lr = 0.01
        args.epochs = 200
        args.decay = '50-100-150'

        args.epochs = 300

    if args.template.find('SESR') >= 0:
        args.model = 'SESR'
        args.patch_size = 128
        # args.optimizer = 'SGD'
        # args.momentum = 0.9
        # args.batch_size = 64
        # args.weight_decay = 0.0001
        args.batch_size = 64
        # args.lr = 0.01
        args.epochs = 200
        args.decay = '50-100-150'

    if args.template.find('PHAMCO') >= 0:
        args.model = 'PHAMCO'
        args.n_resblocks = 10
        args.n_feats = 64
        args.patch_size = 48 * args.scale[0]
        args.epochs = 300




