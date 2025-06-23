



#Benucci Lorenzo, D'Addario Giuseppe, La Sapienza University of Rome, 2025.
#####################################################################################
##### MAIN FUNCTION ACTUALLY MOVED ON COLAB. THIS ONE IS USED ONLY IN LOCAL ENV #####
#####################################################################################
#https://colab.research.google.com/drive/1ixKbLo5EVUr1gbbYYy1jvVuDGWIKiHK7?usp=sharing



## Translations
from src import correct_with_llama4, correct_with_minerva, correct_with_t5

## Scores
from src import gemini_score, prometheus_score, annotate_human_scores, build_output_file

## Stats
from src import kappa_correlation,accuracy_correlation, plot_confusion_matrix, build_rouges,plot_confusion_matrices

# WATCH OUT! this is just the name of the file without extension. it MUST be into the datasets folder.
# The output will be generated into /datasets/correction with the same file_name and extension.
FILE_NAME = "the_vampyre"




#------ TRANSLATIONS ------#

# - llama4 -
# correct_with_llama4(FILE_NAME, print_result=False) ##Enrich the dataset with translations
#
# # - smallLLM -
#correct_with_t5(FILE_NAME, print_result=True)
#
# # - minerva - correction_model is the name of the model to use for translation in:
# [minerva, minerva_finetuned_llima, minerva_finetuned_post_ocr]
# correct_with_minerva(FILE_NAME, correction_model="minerva")
#
#build_output_file(FILE_NAME, "t5")
#build_output_file(FILE_NAME, "llama4")
#build_output_file(FILE_NAME, "minerva")
#build_output_file(FILE_NAME, "minerva_finetuned_llima")
#build_output_file(FILE_NAME, "minerva_finetuned_post_ocr")
# # ------------------------------#
#
#
#
# #------  ANNOTATIONS  ------#
# #function input "corection_model" can be [llama4/minerva/smallLLM]
#
# # - Gemini -correction_model is the name of the model to use for translation in:
# # [llama4, deep_mount, minerva, minerva_finetuned_llima, minerva_finetuned_post_ocr]
#gemini_score(FILE_NAME, "llama4") ## Evaluating llama4 translations
#gemini_score(FILE_NAME,"t5") ## Evaluating smallLLM translations
#gemini_score(FILE_NAME,"minerva") ##Evaluating minerva translations
#gemini_score(FILE_NAME,"minerva_finetuned_llima") ##Evaluating minerva translations
#gemini_score(FILE_NAME,"minerva_finetuned_post_ocr") ##Evaluating minerva translations

#build_rouges(FILE_NAME, "t5")
#build_rouges(FILE_NAME, "minerva")
#build_rouges(FILE_NAME, "minerva_finetuned_llima")
#build_rouges(FILE_NAME, "minerva_finetuned_post_ocr")
#build_rouges(FILE_NAME, "llama4")
#
#
# # - Prometheus -correction_model is the name of the model to use for translation in:
# # # [llama4, deep_mount, minerva, minerva_finetuned_llima, minerva_finetuned_post_ocr]
#prometheus_score(FILE_NAME, "llama4") ## Evaluating llama4 translations
# prometheus_score(FILE_NAME,"deep_mount") ## Evaluating smallLLM translations
# prometheus_score(FILE_NAME,"minerva") ## Evaluating minerva translations.
#
#
# # - Human -correction_model is the name of the model to use for translation in:
# # # [llama4, deep_mount, minerva, minerva_finetuned_llima, minerva_finetuned_post_ocr]
# annotate_human_scores(FILE_NAME,"llama4")
# annotate_human_scores(FILE_NAME,"t5")
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
#kappa_correlation(FILE_NAME,"llama4","gemini",print_results=True)
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
#plot_confusion_matrix("the_vampyre", "llama4", "gemini", normalize=True)
#plot_confusion_matrix("the_vampyre", "llama4", "prometheus", normalize=True)
#
# plot_confusion_matrix("finetuning", "smaLLM", "gemini", normalize=True)
# plot_confusion_matrix("finetuning", "smaLLM", "prometheus", normalize=True)
#
# plot_confusion_matrix("finetuning", "minerva", "gemini", normalize=True)
# plot_confusion_matrix(FILE_NAME, "minerva", "prometheus", normalize=True)

#plot_confusion_matrices("the_vampyre", "llama4", normalize=True)
#plot_confusion_matrices("the_vampyre", "minerva", normalize=True)
#plot_confusion_matrices("the_vampyre", "t5", normalize=True)
