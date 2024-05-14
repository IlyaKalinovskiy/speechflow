include batch.praat

procedure action
	s$ = selected$("Sound")
	runScript: "changepitchmedian.praat", 0, 0, 0
	Rename: s$ + "-monotone"
endproc
