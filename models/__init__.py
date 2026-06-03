from models.convnext import MODEL_ID as CONVNEXT_MODEL_ID
from models.convnext import get_model as get_convnext
from models.convnext import get_processor as get_convnext_processor
from models.convnext_tiny import MODEL_ID as CONVNEXT_TINY_MODEL_ID
from models.convnext_tiny import get_model as get_convnext_tiny
from models.convnext_tiny import get_processor as get_convnext_tiny_processor
from models.dinov2 import MODEL_ID as DINOV2_MODEL_ID
from models.dinov2 import get_model as get_dinov2
from models.dinov2 import get_processor as get_dinov2_processor
from models.convnext_large import MODEL_ID as CONVNEXT_LARGE_MODEL_ID
from models.convnext_large import get_model as get_convnext_large
from models.convnext_large import get_processor as get_convnext_large_processor
from models.resnet50 import MODEL_ID as RESNET50_MODEL_ID
from models.resnet50 import get_model as get_resnet50
from models.resnet50 import get_processor as get_resnet50_processor
from models.resnet101 import MODEL_ID as RESNET101_MODEL_ID
from models.resnet101 import get_model as get_resnet101
from models.resnet101 import get_processor as get_resnet101_processor
from models.vit import MODEL_ID as VIT_MODEL_ID
from models.vit import get_model as get_vit
from models.vit import get_processor as get_vit_processor

MODELS = [
    {
        "name": "dinov2",
        "model_id": DINOV2_MODEL_ID,
        "get_model": get_dinov2,
        "get_processor": get_dinov2_processor,
        "output_dir": "checkpoints/dinov2",
        "freeze_backbone": True,
    },
    {
        "name": "convnext_tiny",
        "model_id": CONVNEXT_TINY_MODEL_ID,
        "get_model": get_convnext_tiny,
        "get_processor": get_convnext_tiny_processor,
        "output_dir": "checkpoints/convnext_tiny",
    },
    {
        "name": "resnet50",
        "model_id": RESNET50_MODEL_ID,
        "get_model": get_resnet50,
        "get_processor": get_resnet50_processor,
        "output_dir": "checkpoints/resnet50",
        "learning_rate": 1e-4,
    },
    {
        "name": "resnet101",
        "model_id": RESNET101_MODEL_ID,
        "get_model": get_resnet101,
        "get_processor": get_resnet101_processor,
        "output_dir": "checkpoints/resnet101",
        "learning_rate": 1e-4,
    },
    {
        "name": "convnext",
        "model_id": CONVNEXT_MODEL_ID,
        "get_model": get_convnext,
        "get_processor": get_convnext_processor,
        "output_dir": "checkpoints/convnext",
        "learning_rate": 3e-4,  # best from LR sweep (config 11: 76.27%)
    },
    {
        "name": "convnext_large",
        "model_id": CONVNEXT_LARGE_MODEL_ID,
        "get_model": get_convnext_large,
        "get_processor": get_convnext_large_processor,
        "output_dir": "checkpoints/convnext_large",
        "learning_rate": 3e-4,  # same family; apply same LR finding
    },
    {
        "name": "vit",
        "model_id": VIT_MODEL_ID,
        "get_model": get_vit,
        "get_processor": get_vit_processor,
        "output_dir": "checkpoints/vit",
    },
]
