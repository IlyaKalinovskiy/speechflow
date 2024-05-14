form Change note
	optionmenu New_note 1
		option C	Do
		option C#	Do sharp
		option D	Re
		option D#	Re sharp
		option E	Mi
		option F	Fa
		option F#	Fa sharp
		option G	Sol
		option G#	Sol sharp
		option A	La
		option A#	La sharp
		option B	Ti
	choice Octave 2
		button C2 to B2
		button C3 to B3
		button C4 to B4
		button C5 to B5
	boolean Monotone 0
	boolean Preview_(click_Apply._Uncheck_to_publish) 1
endform

include batch.praat

procedure action
	s = selected("Sound")
	s$ = selected$("Sound")
	freq = 440 * 2 ^ ((octave - 1) + (new_note - 34) / 12)

	if monotone
		pitch_variation = 0
		m$ = "_monotone"
	else
		pitch_variation = 100
		m$ = ""
	endif

include preview1.inc

	runScript: "changepitchmedian.praat", freq, pitch_variation, 0
	result = selected("Sound")

include preview2.inc

	if not preview
		Rename: s$ + "-changenote_" + extractLine$(new_note$, tab$) + "_Octave_" + string$(octave + 1) + m$
	endif
endproc
