
## Translations
from src import translate_with_llama4, translate_with_minerva, translate_with_deep_mount

## Scores
from src import gemini_score, prometheus_score, annotate_human_scores

## Stats
from src import kappa_correlation,accuracy_correlation, plot_confusion_matrix

# WATCH OUT! this is just the name of the file without extension. it MUST be into the datasets folder.
# The output will be generated into /datasets/correction with the same file_name and extension.
FILE_NAME = "the_vampyre"




#------ TRANSLATIONS ------#

# - llama4 -
# translate_with_llama4(FILE_NAME, print_result=False) ##Enrich the dataset with translations
#
# # - smallLLM -
# translate_with_deep_mount(FILE_NAME, print_result=False)
#
# # - minerva - correction_model is the name of the model to use for translation in:
# [minerva, minerva_finetuned_llima, minerva_finetuned_post_ocr]
# translate_with_minerva(FILE_NAME, correction_model="minerva")
#
# # ------------------------------#
#
#
#
# #------  ANNOTATIONS  ------#
# #function input "corection_model" can be [llama4/minerva/smallLLM]
#
# # - Gemini -correction_model is the name of the model to use for translation in:
# # [llama4, deep_mount, minerva, minerva_finetuned_llima, minerva_finetuned_post_ocr]
# gemini_score(FILE_NAME, "llama4") ## Evaluating llama4 translations
# gemini_score(FILE_NAME,"deep_mount") ## Evaluating smallLLM translations
# gemini_score(FILE_NAME,"minerva") ##Evaluating minerva translations
#
#
# # - Prometheus -correction_model is the name of the model to use for translation in:
# # # [llama4, deep_mount, minerva, minerva_finetuned_llima, minerva_finetuned_post_ocr]
# prometheus_score(FILE_NAME, "llama4") ## Evaluating llama4 translations
# prometheus_score(FILE_NAME,"deep_mount") ## Evaluating smallLLM translations
# prometheus_score(FILE_NAME,"minerva") ## Evaluating minerva translations.
#
#
# # - Human -correction_model is the name of the model to use for translation in:
# # # [llama4, deep_mount, minerva, minerva_finetuned_llima, minerva_finetuned_post_ocr]
# annotate_human_scores(FILE_NAME,"llama4")
# annotate_human_scores(FILE_NAME,"smallLLM")
# annotate_human_scores(FILE_NAME,"minerva")
# # ------------------------------#
#
#
#
#
#
# # ----- COMPUTING CORRELATION ----- #
# -correction_model is the name of the model to use for translation in:
# # [llama4, deep_mount, minerva, minerva_finetuned_llima, minerva_finetuned_post_ocr]
# # kappa_correlation(FILE_NAME, correction_model, judge_model, print_result)
# # accuracy_correlation(FILE_NAME, correction_model, judge_model, print_result)
#
# kappa_correlation(FILE_NAME,"llama4","gemini",print_results=True)
# kappa_correlation(FILE_NAME,"smallLLM","gemini",print_results=True)
# kappa_correlation(FILE_NAME,"minerva","gemini",print_results=True)
#
# kappa_correlation(FILE_NAME,"llama4","prometheus",print_results=True)
# kappa_correlation(FILE_NAME,"smallLLM","prometheus",print_results=True)
# kappa_correlation(FILE_NAME,"minerva","prometheus",print_results=True)
#
#
# accuracy_correlation(FILE_NAME,"llama4","gemini",print_results=True)
# accuracy_correlation(FILE_NAME,"smallLLM","gemini",print_results=True)
# accuracy_correlation(FILE_NAME,"minerva","gemini",print_results=True)
#
# accuracy_correlation(FILE_NAME,"llama4","prometheus",print_results=True)
# accuracy_correlation(FILE_NAME,"smallLLM","prometheus",print_results=True)
# accuracy_correlation(FILE_NAME,"minerva","prometheus",print_results=True)
# # ------------------------------#
#
#
# ## -------- CONFUSION MATRICES -------- #
# plot_confusion_matrix("finetuning", "llama4", "gemini", normalize=True)
# plot_confusion_matrix("finetuning", "llama4", "prometheus", normalize=True)
#
# plot_confusion_matrix("finetuning", "smaLLM", "gemini", normalize=True)
# plot_confusion_matrix("finetuning", "smaLLM", "prometheus", normalize=True)
#
# plot_confusion_matrix("finetuning", "minerva", "gemini", normalize=True)
# plot_confusion_matrix(FILE_NAME, "minerva", "prometheus", normalize=True)