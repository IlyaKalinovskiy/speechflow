include batch.praat

procedure action
	s$ = selected$("Sound")
	runScript: "changepitchmedian.praat", 0, -100, 0
	Rename: s$ + "-invertpitch"
endproc
