from backend.config import settings
from backend.schemas import Routing

class ConfidenceRouter:
    def route(self, prediction_label: str, confidence: float, ood_flag: bool = False) -> Routing:
        """
        Routes the document depending on probability logic, out of distribution 
        results, and thresholds.
        """
        if ood_flag:
            return Routing.OOD
            
        if confidence >= settings.auto_approve_threshold:
            return Routing.DIRECT
            
        if confidence >= settings.uncertain_threshold:
            return Routing.ASYNC_CONFIRM
            
        return Routing.HUMAN_REVIEW
