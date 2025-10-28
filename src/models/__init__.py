# Lazy imports to avoid circular import issues
def __getattr__(name):
    if name == "Baseline":
        from .baseline import Baseline
        return Baseline
    elif name == "BaselineMaskedRetrievalHead":
        from .baseline_masked_retrieval_head import BaselineMaskedRetrievalHead
        return BaselineMaskedRetrievalHead
    elif name == "BaselineMaskedNonRetrievalHead":
        from .baseline_masked_non_retrieval_head import BaselineMaskedNonRetrievalHead
        return BaselineMaskedNonRetrievalHead
    elif name == "ContrastiveDecoding":
        from .contrastive_decoding import ContrastiveDecoding
        return ContrastiveDecoding
    elif name == "DeCoReVanilla":
        from .decore_vanilla import DeCoReVanilla
        return DeCoReVanilla
    elif name == "DeCoReEntropy":
        from .decore_entropy import DeCoReEntropy
        return DeCoReEntropy
    elif name == "DeCoReEntropyGain":
        from .decore_entropy_gain import DeCoReEntropyGain
        return DeCoReEntropyGain
    elif name == "DoLa":
        from .dola import DoLa
        return DoLa
    elif name == "DeCoReRandomEntropy":
        from .decore_random_entropy import DeCoReRandomEntropy
        return DeCoReRandomEntropy
    elif name == "ActivationDecoding":
        from .activation_decoding import ActivationDecoding
        return ActivationDecoding
    elif name == "ContextAwareDecoding":
        from .context_aware_decoding import ContextAwareDecoding
        return ContextAwareDecoding
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
