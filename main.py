from src.correlations import kappa_correlation,accuracy_correlation
from src.gemini_score import gemini_score
from src.human_score import annotate_human_scores
from src.llama4_translation import translate_with_llama4








####### LLAMA4 TRANSLATIONS ########
FILE_PATH = r"datasets\eng\finetuning.json"
translate_with_llama4(FILE_PATH)



#######   GEMINI ANNOTATING  ########
FILE_PATH = r"datasets\eng\corrections\finetuning_correction.json"
gemini_score(FILE_PATH)



#######   HUMAN ANNOTATING  ########
FILE_PATH = r"datasets\eng\corrections\finetuning_correction.json"
annotate_human_scores(FILE_PATH)



####### COMPUTING CORRELATION #########
FILE_PATH = r"datasets\eng\corrections\finetuning_correction.json"
kappa_correlation(FILE_PATH,True)
accuracy_correlation(FILE_PATH,True)