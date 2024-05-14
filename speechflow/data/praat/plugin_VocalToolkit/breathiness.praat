form Breathiness
	real Breathiness_(%) 25
	boolean Preview_(click_Apply._Uncheck_to_publish) 1
endform

breathiness = min(max(breathiness, 0), 100)

include batch.praat

procedure action
	s = selected("Sound")
	s$ = selected$("Sound")

include preview1.inc

	runScript: "whisper.praat"
	whisper = selected("Sound")

	if preview
		plusObject: pre1
	else
		plusObject: s
	endif

	runScript: "copymix.praat", breathiness, 1, 0
	runScript: "fixdc.praat"
	result = selected("Sound")

	removeObject: whisper

include preview2.inc

	if not preview
		Rename: s$ + "-breathiness_" + string$(breathiness)
	endif
endproc
