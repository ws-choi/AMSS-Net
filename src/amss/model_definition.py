from src.amss.cunet.models.dcun_isolasional_pocm import DCUN_Isolasion_PoCM_Framework
from src.amss.cunet.models.dcun_isolasional_smpocm import DCUN_Isolasion_SMPoCM_Framework
from src.amss.cunet.models.dcun_lasaft_gpocm import DCUN_TFC_GPoCM_LaSAFT_Framework
from src.amss.cunet.models.dcun_lasaft_smpocm import DCUN_LaSAFT_SMPoCM_Framework
from src.amss.cunet.models.iasmnet_top_csa import IaSMNet_Top_CSA_Framework


def get_class_by_name(model_name):
    if model_name == 'lasaft_gpocm':
        return DCUN_TFC_GPoCM_LaSAFT_Framework
    elif model_name == 'isolasion_smpocm':
        return DCUN_Isolasion_SMPoCM_Framework
    elif model_name == 'lasaft_smpocm':
        return DCUN_LaSAFT_SMPoCM_Framework
    elif model_name == 'isolasion_pocm':
        return DCUN_Isolasion_PoCM_Framework
    elif model_name == 'iasmnet_top_csa':
        return IaSMNet_Top_CSA_Framework

    # elif model_name == 'CUNET_TFC_SPoCM_LightSAFT':
    #     return DCUN_TFC_SPoCM_LightSAFT_Framework
    # elif model_name == 'CUNET_TFC_SPoCM_LaSAFT':
    #     return DCUN_TFC_SPoCM_LaSAFT_Framework
    # elif model_name == 'CAUNET_DTFC_SMPoCM_LightSAFT':
    #     return DCAUN_DTFC_SMPoCM_LightSAFT_Framework
    else:
        raise NotImplementedError

#######################
