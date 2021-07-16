from text.pinyin import symbols
from text.speakers import speakers

class hparams:
        ################################
        # Experiment Parameters        #
        ################################
        epochs = 300
        iters_per_checkpoint = 10000
        iters_per_validation = 1000
        seed = 4321
        dynamic_loss_scaling = True
        distributed_run = False
        cudnn_enabled = True
        cudnn_benchmark = False
        ignore_layers = ['embedding.weight']
        speaker_embedding_type = 'local'   # one-hot, global, local, local_vae, gst, vae
        ref_mel_type = 'unpair'   # pair, unpair

        ################################
        # Data Parameters             #
        ################################
        mel_training_files = './training_data/mel-aishell3_character_pinyin_data_train.txt'
        mel_validation_files = './training_data/mel-aishell3_character_pinyin_data_val.txt'
        melpath_prefix = '/data/datasets/aishell3_train/mels/' # If precise mel-spectrogram path is provided in above files, this param is set to None.
        text_cleaners = ['basic_cleaners']

        ################################
        # Model Parameters             #
        ################################
        mel_dim = 80
        num_symbols = len(symbols)
        text_embedding_dim = 512
        stop_threshold = 0.5
        n_frames_per_step = 3
        max_decoder_steps = 1000

        # speaker embedding
        num_speakers = len(speakers)
        speaker_embedding_dim = 128
        speaker_loss_weight = 0.0
        spk_classifier_hidden_dims = [256]

        # Reference Encoder parameters
        ref_conv_channels = [32, 32, 64, 64, 128, 128]
        ref_conv_stride = [2, 1, 2, 1, 2, 1]
        ref_gru_units = 128
        ref_local_style_dim = 8
        ref_attention_dropout = 0.0
        ref_attention_dim = 128
        ref_speaker_loss_weight = 0.1

        # GST parameters
        gst_num_tokens = 1024
        gst_num_heads = 8
        gst_conv_channels = [32, 32, 64, 64, 128, 128]
        gst_gru_units = 128
        gst_token_dim = 256
        gst_num_units = 256

        # VAE parameters
        vae_latent_dim = 32
        vae_out_dim = 256
        anneal_function = 'constant'
        anneal_k = 0.0025
        anneal_x0 = 10000
        anneal_upper = 0.2
        anneal_lag = 50000

        # Encoder parameters
        encoder_num_convs = 3
        encoder_conv_kernel_size = 5
        encoder_conv_channels = 512
        encoder_conv_dropout = 0.5
        encoder_blstm_units = 512

        # Decoder parameters
        prenet_dims = [256, 256]
        prenet_dropout = 0.5
        # Attention parameters
        attention_location_filters = 32
        attention_location_kernel_size = 31
        attention_dim = 128
        attention_rnn_units = 1024
        attention_dropout = 0.1
        decoder_rnn_units = 1024
        decoder_dropout = 0.1

        p_attention_dropout=0.1

        # Mel-post processing network parameters
        postnet_num_convs = 5
        postnet_conv_channels = 512
        postnet_conv_kernel_size = 5
        postnet_conv_dropout = 0.5


        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate = False
        learning_rate = 1e-3
        weight_decay = 1e-6
        grad_clip_thresh = 1.0
        batch_size = 32
        mask_padding = True  # set model's padded outputs to padded values

