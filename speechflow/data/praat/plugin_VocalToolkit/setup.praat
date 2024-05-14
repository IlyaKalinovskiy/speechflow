# Praat Vocal Toolkit
# A Praat plugin with automated scripts for voice processing
# https://www.praatvocaltoolkit.com
# This plugin is open source and can be used for any purpose

if praatVersion < 6104
	beginPause: "Praat Vocal Toolkit - Unsupported Praat version"
		comment: "“Praat Vocal Toolkit” requires Praat version 6.1.04 or higher."
		comment: "Your version of Praat is " + praatVersion$ + ". Please update it at ""https://www.praat.org""."
	endPause: "OK", 1, 1
else
	runScript: "buttons.praat"
	runScript: "updatepresetslists.praat"
endif
