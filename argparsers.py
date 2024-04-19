def model_train_parser(parser):
    """Stores arguments for train.py argparser"""
    parser.add_argument(
        "--wandb_proj",
        type=str,
        default="same-different-transformers",
        help="Name of WandB project to store the run in.",
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="Team to send run to."
    )
    parser.add_argument(
        "--num_gpus", type=int, help="number of available GPUs.", default=1
    )

    # Model/architecture arguments
    parser.add_argument(
        "-m",
        "--model_type",
        help="Model to train: vit, clip_vit.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--patch_size", type=int, default=32, help="Size of patch (eg. 16 or 32)."
    )
    parser.add_argument(
        "--auxiliary_loss",
        action="store_true",
        default=False,
        help="Train with auxiliary loss to induce subspaces.",
    )

    parser.add_argument(
        "--probe_layer", default=-1, type=int, help="Probe layer for auxiliary loss"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=False,
        help="Use ImageNet pretrained models. If false, models are trained from scratch.",
    )

    # Training arguments
    parser.add_argument(
        "-ds",
        "--dataset_str",
        required=False,
        help="Name of the directory containing stimuli",
        default="NOISE_RGB",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw",
        help="Training optimizer, eg. adam, adamw, sgd.",
    )
    parser.add_argument("--lr", type=float, default=2e-6, help="Learning rate.")
    parser.add_argument(
        "--lr_scheduler", default="reduce_on_plateau", help="LR scheduler."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=30, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Train/validation batch size."
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="If not given, picks random seed."
    )

    # Dataset arguments
    parser.add_argument(
        "--k", 
        type=int, 
        default=5, 
        help="Number of objects per scene for counting task."
    )
    parser.add_argument(
        "--n_images_per_task",
        type=int,
        default=100,
        help="Number of images per counting task (per class).",
    )
    
    parser.add_argument(
        "--compositional",
        type=int,
        default=-1,
        help="Create compositional NOISE_RGB dataset with specified # of combinations in train set.",
    )
    parser.add_argument(
        "--n_train", type=int, default=6400, help="Size of training dataset to use."
    )
    parser.add_argument(
        "--n_train_tokens",
        type=int,
        default=256,
        help="Number of unique tokens to use \
                        in the training dataset. If -1, then the maximum number of tokens is used.",
    )
    parser.add_argument(
        "--n_val_tokens",
        type=int,
        default=256,
        help="Number of unique tokens to use \
                        in the validation dataset. If -1, then number tokens = (total - n_train_tokens) // 2.",
    )
    parser.add_argument(
        "--n_test_tokens",
        type=int,
        default=256,
        help="Number of unique tokens to use \
                        in the test dataset. If -1, then number tokens = (total - n_train_tokens) // 2.",
    )
    parser.add_argument(
        "--n_val",
        type=int,
        default=6400,
        help="Total # validation stimuli. Default: equal to n_train.",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=6400,
        help="Total # test stimuli. Default: equal to n_train.",
    )
    parser.add_argument(
        "--ood",
        action="store_true",
        help="Whether or not to run OOD evaluations.",
        default=False,
    )
    parser.add_argument(
        "--texture",
        action="store_true",
        default=False,
        help="Create dataset with textures.",
    )
    parser.add_argument(
        "--disentangled_color",
        action="store_true",
        default=False,
        help="Create dataset with color as an additional third axis of variation (disentangled from texture).",
    )

    # Paremeters for logging, storing models, etc.
    parser.add_argument(
        "--save_model_freq",
        help="Number of times to save model checkpoints \
                        throughout training. Saves are equally spaced from 0 to num_epoch.",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--checkpoint",
        help="Whether or not to store model checkpoints.",
        default=True,
    )
    parser.add_argument(
        "--wandb_cache_dir",
        help="Directory for WandB cache. May need to be cleared \
                        depending on available storage in order to store artifacts.",
        default=None,
    )

    parser.add_argument(
        "--wandb_run_dir",
        help="Directory where WandB runs should be stored.",
        default=None,
    )
    
    return parser.parse_args()


def model_probe_parser(parser):
    parser.add_argument(
        "-m",
        "--model_type",
        help="Model to train: vit, clip_vit.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--patch_size", type=int, default=32, help="Size of patch (eg. 16 or 32)."
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Path to state dict to probe.",
    )

    parser.add_argument(
        "-ds",
        "--dataset_str",
        required=False,
        help="Names of the directory containing stimuli",
        default="NOISE_RGB/aligned/N_32/trainsize_6400_32-32-224",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw",
        help="Training optimizer, eg. adam, adamw, sgd.",
    )
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
    parser.add_argument(
        "--lr_scheduler", default="reduce_on_plateau", help="LR scheduler."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=30, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Train/validation batch size."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="If not given, picks random seed."
    )
    args = parser.parse_args()
    return args


def data_generation_parser(parser):
    """Stores arguments for data.py parser"""
    parser.add_argument(
        "--patch_size", type=int, default=32, help="Size of patch (eg. 16 or 32)."
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=6400,
        help="Total # of training stimuli. eg. if n_train=6400, a dataset"
        "will be generated with 3200 same and 3200 different stimuli.",
    )
    parser.add_argument(
        "--n_val",
        type=int,
        default=-1,
        help="Total # validation stimuli. Default: equal to n_train.",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=-1,
        help="Total # test stimuli. Default: equal to n_train.",
    )
    parser.add_argument(
        "--n_train_tokens",
        type=int,
        default=256,
        help="Number of unique tokens to use \
                        in the training dataset. If -1, then the maximum number of tokens is used.",
    )
    parser.add_argument(
        "--n_val_tokens",
        type=int,
        default=256,
        help="Number of unique tokens to use \
                        in the validation dataset. If -1, then number tokens = (total - n_train_tokens) // 2.",
    )
    parser.add_argument(
        "--n_test_tokens",
        type=int,
        default=256,
        help="Number of unique tokens to use \
                        in the test dataset. If -1, then number tokens = (total - n_train_tokens) // 2.",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Folder to get stimuli from inside of the `source` folder",
        default="NOISE_RGB",
    )
    parser.add_argument(
        "--create_source",
        action="store_true",
        default=False,
        help="Create NOISE_RGB source objects (rather than a same-different dataset).",
    )
    parser.add_argument(
        "--compositional",
        type=int,
        default=-1,
        help="Create compositional NOISE_RGB dataset with specified # of combinations in train set.",
    )
    parser.add_argument(
        "--texture",
        action="store_true",
        default=False,
        help="Create dataset with textures.",
    )
    parser.add_argument(
        "--disentangled_color",
        action="store_true",
        default=False,
        help="Create dataset with color as an additional third axis of variation (disentangled from texture).",
    )

    return parser.parse_args()
