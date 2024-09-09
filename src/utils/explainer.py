from src.abstract.explainer import Explainer


def get_node_explainer(name: str)->Explainer:

    if name == "cf-gnn":
        from src.node_level_explainer import CFExplainer
        return CFExplainer
    
    elif name == "cf-gnnfeatures":
        from src.node_level_explainer import CFExplainerFeatures
        return CFExplainerFeatures

    elif name == "random":
        from src.node_level_explainer import RandomExplainer
        return RandomExplainer

    elif name == "random-feat":
        from src.node_level_explainer import RandomFeaturesExplainer
        return RandomFeaturesExplainer
    
    elif name == "ego":
        from src.node_level_explainer import EgoExplainer
        return EgoExplainer
    
    elif name == "cff":
        from src.node_level_explainer import CFFExplainer
        return CFFExplainer

    elif name == "gnn-explainer":
        raise NotImplemented("Not implemented yet!")
    
    else:
        raise ValueError(f"Technique not implemented {name}")