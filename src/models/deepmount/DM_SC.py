# DEEPMount + SpellChecker (JUST ITALIAN!!)


from transformers import pipeline
from spellchecker import SpellChecker

spell = SpellChecker(language='en')
ocr_corrector = pipeline("text2text-generation", model="DeepMount00/OCR_corrector")

ocr_text = "THEsuperstition upon which this taIe iſ founded is very general in the East. Among tho Arabjans itappeors to be common: it did not, however, extend itself to the Greeks until after the establi shment of Christianity; and it has only aſsumed its prosent form since the division af the Latin and Greok churches; at which time, lhe idea becoming prevalent, that a Lcltin body could not corrvpl if buried in their territory, it gradually increosed, and formed lhe subject of many wonderful stories, ſtill extant, of the dead rising from their graves, and feeding uponlhe blood of tho young and beautiful. In the West itspread, with some slight variation, all over Hungary, Poland, Austria, and Lorraine, whoro the helies existed, that vompyresnightly imbi6ed a certain portion of the blood of their victims, who became emaciated, lost their strength, and speedily died of c0nsumptions; whilst these human blood-suckers fattened—and their veins became distended to such a state of ropletion, as t0 cause the blood to flow from all the passages of their badies, and even fr0m the ucry pores of thoir skins."


#ocr_text = "\n\n\n\u2014  Bada,  Pinocchio!...  il  mostro  ti  raggiunge! \nEccolo!...  Eccolo!...  Affrettati,  per  carit\u00e0,  o  sei \nperduto  !...  \u2014 \nE  Pinocchio  a  nuotare  pi\u00f9  lesto  che  mai,  e  via, \nvia,  e  via,  come  anderebbe  una  palla  di  fucile. \n\nr \n\n'%^^--^^ \n\nE  Pinocchio  nuotava  disperatamente  con  le  braccia,  col  petto, \ncon  le  gambe  e  coi  piedi. \n\nE  gi\u00e0  si  accostava  allo  scoglio,  e  gi\u00e0  la  capret- \ntina  spenzolandosi  tutta  sul  mare,  gli  porgeva \nle  sue  zampine  davanti  per  aiutarlo  a  uscir  fuori \ndell'acqua....  Ma!... \nMa  oramai  era  tardi  !  Il  mostro  lo  aveva  rag- \ngiunto. Il  mostro,  tirando  il  fiato  a  s\u00e9,  si  bevve \n\nil  povero  burattino,  come  avrebbe  bevuto  un \nuovo  di  gallina,  e  lo  inghiott\u00ec  con  tanta  violenza \ne  con  tanta  avidit\u00e0,  che  Pinocchio,  cascando  gi\u00f9 \nin  corpo  al  Pesce-cane,  batt\u00e8  un  colpo  cos\u00ec  screan- \nzato da  restarne  sbalordito  per  un  quarto  d' ora. \nQuando  ritorn\u00f2  in  s\u00e9  da  quello  sbigottimento, \nnon  sapeva  raccapezzarsi,  nemmeno  lui,  in  che \nmondo  si  fosse.  Intorno  a  se  e'  era  da  ogni  parte \nun  gran  buio  :  ma  un  buio  cos\u00ec  nero  e  profondo, \nche  gli  pareva  di  essere  entrato  col  capo  in  un \ncalamaio  pieno  d'inchiostro.  Stette  in  ascolto  e \nnon  sent\u00ec  nessun  rumore;  solamente  di  tanto  in \ntanto  sentiva  battersi  nel  viso  alcune  grandi  buf- \nfate di  vento.  Da  principio  non  sapeva  intendere \nda  dove  quel  vento  uscisse:  ma  poi  cap\u00ec  che \nusciva  dai  polmoni  del  mostro.  Perch\u00e8  bisogna  sa- \npere che  il  Pesce-cane  soffriva  moltissimo  d'asma, \ne  quando  respirava  pareva  proprio  che  solliasse \nla  tramontana. \nPinoccliio,  sulle  prime,  s' ingegn\u00f2  di  farsi  un \npo'  di  coraggio  :  ma  quand'  ebbe  la  prova  e  la \nriprova  di  trovarsi  chiuso  in  corpo  al  mostro  ma- \nrino allora  cominci\u00f2  a  piangere  e  a  strillare;  e \npiangendo  diceva: \n\u2014  Aiuto!  aiuto!  Oli  povero  me!  Non  e'  \u00e8  nes- \nsuno che  venga  a  salvarmi! \n\u2014  Ohi  vuoi  che  ti  salvi,  disgraziato  !..  \u2014  disse \n\n\nin "




# Sostituzioni preliminari
preprocessed = ocr_text.replace('1', 'i').replace('0', 'o').replace('4', 'a')

# Correzione modello
corrected = ocr_corrector(preprocessed)[0]['generated_text']

# Correzione spellchecker finale parola per parola
words = corrected.split()
final_words = [spell.correction(w) or w for w in words]
final_text = " ".join(final_words)

print("Testo finale corretto:", final_text)
