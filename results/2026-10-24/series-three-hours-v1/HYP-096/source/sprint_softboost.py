"""H80: new combination of H72 exact count-F1 gradient with residual trees."""
import numpy as np
from scipy.special import softmax
from sprint_ambitious import LovaszTeacherBoost
from sprint_classification import soft_macro_f1_loss_and_gradient


class MacroTeacherBoost(LovaszTeacherBoost):
    def __init__(self,iterations=300,threads=2):
        super().__init__(iterations=iterations,threads=threads)
    def configuration(self):
        result=super().configuration()
        result.update(hypothesis='HYP-080',family='teacher_residual_soft_macro_f1_lightgbm',
                      objective='.5meanCE+.5(1-softMacroF1)',gradient='exact_fullbatch_coupled_gradient_times_N',
                      curvature='positive_surrogate_max(.5*p*(1-p),.001)_not_exact')
        return result
    def _objective(self,labels,logits):
        y=np.asarray(labels,np.int64);z=np.asarray(logits,np.float64)
        loss,gradient=soft_macro_f1_loss_and_gradient(z,y)
        self.loss_history_.append(loss)
        p=softmax(z,axis=1)
        return len(y)*gradient,np.maximum(.5*p*(1-p),.001)
