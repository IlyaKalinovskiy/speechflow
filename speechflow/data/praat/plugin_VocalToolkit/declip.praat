include batch.praat

procedure action
	extr = Get absolute extremum: 0, 0, "None"

	if extr = undefined
		extr = 0
	endif

	if number(fixed$(extr, 2)) > 0.99
		Scale peak: 0.99
	endif
endproc
