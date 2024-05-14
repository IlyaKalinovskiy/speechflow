# EQ impulse responses list

Create Strings as file list: "list", "eq/*.Sound"
nFiles_Eq = Get number of strings
for i to nFiles_Eq
	fileNameEq$[i] = Get string: i
endfor
Remove

newEqList$ = ""
for i to nFiles_Eq
	l = length(fileNameEq$[i]) - 6
	fileNameEq$ = left$(fileNameEq$[i], l)
	newEqList$ = newEqList$ + "option " + fileNameEq$ + newline$
endfor

eqListFile$ = "eqpresetslist.inc"
if fileReadable (eqListFile$)
	currentEqList$ = readFile$ (eqListFile$)
	if newEqList$ <> currentEqList$
		writeFile: "eqpresetslist.inc", newEqList$
	endif
else
	writeFile: "eqpresetslist.inc", newEqList$
endif


# Reverb impulse responses list

Create Strings as file list: "list", "reverb/*.wav"
nFiles_Reverb = Get number of strings
for i to nFiles_Reverb
	fileNameReverb$[i] = Get string: i
endfor
Remove

newReverbList$ = ""
for i to nFiles_Reverb
	l = length(fileNameReverb$[i]) - 4
	fileNameReverb$ = left$(fileNameReverb$[i], l)
	newReverbList$ = newReverbList$ + "option " + fileNameReverb$ + newline$
endfor

reverbListFile$ = "reverbpresetslist.inc"
if fileReadable (reverbListFile$)
	currentReverbList$ = readFile$ (reverbListFile$)
	if newReverbList$ <> currentReverbList$
		writeFile: "reverbpresetslist.inc", newReverbList$
	endif
else
	writeFile: "reverbpresetslist.inc", newReverbList$
endif


# TTS languages and voices list

languages = Extract espeak data: "Language properties"
numberOfLanguages = Get number of rows
for i to numberOfLanguages
	language$[i] = Get value: i, "name"
endfor
default_language = Search column: "name", "English (Great Britain)"
Remove

voices = Extract espeak data: "Voices properties"
numberOfVoices = Get number of rows
for i to numberOfVoices
	voice$[i] = Get value: i, "name"
endfor
default_voice = Search column: "name", "Female1"
Remove

languagesList$ = ""
for i to numberOfLanguages
	languagesList$ = languagesList$ + "option " + language$[i] + newline$
endfor

voicesList$ = ""
for i to numberOfVoices
	voicesList$ = voicesList$ + "option " + voice$[i] + newline$
endfor

newTTSList$ = "optionmenu Language " + string$(default_language) + newline$ + languagesList$ + "optionmenu Voice " + string$(default_voice) + newline$ + voicesList$

ttsListFile$ = "ttspresetslist.inc"
if fileReadable (ttsListFile$)
	currentTTSList$ = readFile$ (ttsListFile$)
	if newTTSList$ <> currentTTSList$
		writeFile: "ttspresetslist.inc", newTTSList$
	endif
else
	writeFile: "ttspresetslist.inc", newTTSList$
endif
