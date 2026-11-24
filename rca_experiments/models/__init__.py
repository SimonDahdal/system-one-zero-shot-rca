"""Model registry. Imports are lazy so RF does not require the KEV SDK."""
from importlib import import_module

MODEL_REGISTRY = {
    "rf": ("rca_experiments.models.random_forest", "RandomForestModel"),
    "laya": ("rca_experiments.models.laya", "LayaModel"),
    "nimble": ("rca_experiments.models.nimble", "NimbleModel"),
    "semif": ("rca_experiments.models.semif", "SemIfModel"),
    "rizzo_flow": ("rca_experiments.models.rizzo_flow", "RizzoFlowModel"),
    "von": ("rca_experiments.models.von", "VonModel"),
    "nanojev": ("rca_experiments.models.nanojev", "NanoJevModel"),
    "LiquidAI_d1_3B": ("rca_experiments.models.liquidai_d1_3b", "LiquidAID1Model"),
    "decider": ("rca_experiments.models.decider", "DeciderModel"),
    "h2o_lightning_4B": ("rca_experiments.models.h2o_lightning_4b", "H2OLightningModel"),
    "neohorse": ("rca_experiments.models.neohorse", "NeoHorseModel"),
    "d3_flash_server": ("rca_experiments.models.d3_flash_server", "D3FlashModel"),
    "kev": ("rca_experiments.models.kev", "KEVModel"),
}


def create_model(name):
    module, class_name = MODEL_REGISTRY[name]
    return getattr(import_module(module), class_name)()
