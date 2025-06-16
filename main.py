from src.correlations import kappa_correlation,accuracy_correlation
from src.gemini_score import gemini_score
from src.human_score import annotate_human_scores
from src.llama4_translation import translate_with_llama4




# WATCH OUT! this is just the name of the file without extension. it MUST be into the datasets folder. 
# The output will be generated into /datasets/correction with the same file_name and extension.
FILE_NAME = "finetuning"





####### LLAMA4 TRANSLATIONS ########
translate_with_llama4(FILE_NAME)



#######   GEMINI ANNOTATING  ########
gemini_score(FILE_NAME)



#######   HUMAN ANNOTATING  ########
annotate_human_scores(FILE_NAME)



####### COMPUTING CORRELATION #########
kappa_correlation(FILE_NAME,True)
accuracy_correlation(FILE_NAME,True)