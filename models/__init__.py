from models.convnext import MODEL_ID as CONVNEXT_MODEL_ID
from models.convnext import get_model as get_convnext
from models.convnext import get_processor as get_convnext_processor
from models.resnet import MODEL_ID as RESNET_MODEL_ID
from models.resnet import get_model as get_resnet
from models.resnet import get_processor as get_resnet_processor

MODELS = [
    {
        "name": "resnet",
        "model_id": RESNET_MODEL_ID,
        "get_model": get_resnet,
        "get_processor": get_resnet_processor,
        "output_dir": "models/resnet",
    },
    {
        "name": "convnext",
        "model_id": CONVNEXT_MODEL_ID,
        "get_model": get_convnext,
        "get_processor": get_convnext_processor,
        "output_dir": "models/convnext",
    },
]
