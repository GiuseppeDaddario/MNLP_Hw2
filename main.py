
## Translations
from src.llama4_translation import translate_with_llama4

## Scores
from src.gemini_score import gemini_score
from src.prometheus_score import prometheus_score
from src.human_score import annotate_human_scores

## Stats
from src.correlations import kappa_correlation,accuracy_correlation





# WATCH OUT! this is just the name of the file without extension. it MUST be into the datasets folder. 
# The output will be generated into /datasets/correction with the same file_name and extension.
FILE_NAME = "finetuning"




#------ TRANSLATIONS ------#

# - llama4 - 
translate_with_llama4(FILE_NAME, print_result=False) ##Enrich the dataset with translations

# - smallLLM - 
#translate_with_smallLLM(FILE_NAME,print_result=False)

# - minerva - 
#translate_with_minerva(FILE_NAME,print_result=False)

# ------------------------------#




#------  ANNOTATIONS  ------#
#function input "corection_model" can be [llama4/minerva/smallLLM]

# - Gemini -
gemini_score(FILE_NAME, "llama4") ## Evaluating llama4 translations
#gemini_score(FILE_NAME,"smallLLM") ## Evaluating smallLLM translations
#gemini_score(FILE_NAME,"minerva") ##Evaluating minerva translations


# - Prometheus - 
prometheus_score(FILE_NAME, "llama4") ## Evaluating llama4 translations
#prometheus_score(FILE_NAME,"smallLLM") ## Evaluating smallLLM translations
#prometheus_score(FILE_NAME,"minerva") ##Evaluating minerva translations


# - Human -
annotate_human_scores(FILE_NAME)

# ------------------------------#





# ----- COMPUTING CORRELATION ----- #

kappa_correlation(FILE_NAME,"llama4","gemini",print_results=True)
#kappa_correlation(FILE_NAME,"smallLLM","gemini",print_results=True)
#kappa_correlation(FILE_NAME,"minerva","gemini",print_results=True)

kappa_correlation(FILE_NAME,"llama4","prometheus",print_results=True)
#kappa_correlation(FILE_NAME,"smallLLM","prometheus",print_results=True)
#kappa_correlation(FILE_NAME,"minerva","prometheus",print_results=True)


accuracy_correlation(FILE_NAME,"llama4","gemini",print_results=True)
#accuracy_correlation(FILE_NAME,"smallLLM","gemini",print_results=True)
#accuracy_correlation(FILE_NAME,"minerva","gemini",print_results=True)

accuracy_correlation(FILE_NAME,"llama4","prometheus",print_results=True)
#accuracy_correlation(FILE_NAME,"smallLLM","prometheus",print_results=True)
#accuracy_correlation(FILE_NAME,"minerva","prometheus",print_results=True)
# ------------------------------#