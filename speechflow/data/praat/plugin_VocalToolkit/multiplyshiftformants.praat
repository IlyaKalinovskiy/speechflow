form Multiply or shift formants
	choice Method 1
		button Multiply (factor)
		button Shift (add Hz)
	real Multiply_or_shift_F1_by 1.1
	real Multiply_or_shift_F2_by 1.1
	real Multiply_or_shift_F3_by 1.1
	real Multiply_or_shift_F4_by 1
	real Multiply_or_shift_F5_by 1
	comment Formant determination
	positive Maximum_formant_(Hz) 5500 (= adult female)
	comment Set 5000 Hz for men, 5500 Hz for women or up to 8000 Hz for children.
	boolean Process_only_voiced_parts 1
	boolean Retrieve_intensity_contour 1
	boolean Preview_(click_Apply._Uncheck_to_publish) 1
endform

f1 = multiply_or_shift_F1_by
f2 = multiply_or_shift_F2_by
f3 = multiply_or_shift_F3_by
f4 = multiply_or_shift_F4_by
f5 = multiply_or_shift_F5_by

include batch.praat

procedure action
	s = selected("Sound")
	s$ = selected$("Sound")
	original_dur = Get total duration
	int = Get intensity (dB)

	if int <> undefined

include preview1.inc

		if process_only_voiced_parts
			@extractUV
			selectObject: extractUV.s_v
			int_v = Get intensity (dB)
			if int_v <> undefined
				int = int_v
			else
				process_only_voiced_parts = 0
				selectObject: s
				removeObject: extractUV.s_u, extractUV.s_v
			endif
		endif

		runScript: "workpre.praat"
		wrk = selected("Sound")
		sf1 = Get sampling frequency

		hf = Filter (pass Hann band): maximum_formant, 0, 100

		selectObject: wrk
		sf2 = maximum_formant * 2
		rs1 = Resample: sf2, 10

		formant = noprogress nowarn To Formant (robust): 0.005, 5, maximum_formant, 0.025, 50, 1.5, 5, 0.000001

		lpc1 = noprogress To LPC: sf2
		plusObject: rs1
		source = Filter (inverse)

		selectObject: formant
		filtr = Copy: "filtr"

		if method = 1
			Formula (frequencies): "if row = 1 then self * f1 else self fi"
			Formula (frequencies): "if row = 2 then self * f2 else self fi"
			Formula (frequencies): "if row = 3 then self * f3 else self fi"
			Formula (frequencies): "if row = 4 then self * f4 else self fi"
			Formula (frequencies): "if row = 5 then self * f5 else self fi"
		else
			Formula (frequencies): "if row = 1 then self + f1 else self fi"
			Formula (frequencies): "if row = 2 then self + f2 else self fi"
			Formula (frequencies): "if row = 3 then self + f3 else self fi"
			Formula (frequencies): "if row = 4 then self + f4 else self fi"
			Formula (frequencies): "if row = 5 then self + f5 else self fi"
		endif

		lpc2 = noprogress To LPC: sf2
		plusObject: source
		tmp = Filter: "no"

		rs2 = Resample: sf1, 10
		Formula: "self + object[hf]"

		runScript: "workpost.praat", original_dur
		Scale intensity: int
		runScript: "declip.praat"

		if process_only_voiced_parts
			@mixUV
		endif

		if retrieve_intensity_contour
			tmp3 = selected("Sound")
			if preview
				plusObject: pre1
			else
				plusObject: s
			endif
			runScript: "copyintensitycontour.praat"
			removeObject: tmp3
		endif
		dur = Get total duration
		if dur > 0.5
			Fade in: 0, 0, 0.005, "yes"
			Fade out: 0, dur, -0.005, "yes"
		endif
		result = selected("Sound")

		removeObject: wrk, hf, rs1, formant, lpc1, source, filtr, lpc2, tmp, rs2

include preview2.inc

		if not preview
			Rename: s$ + "-shiftformants"
		endif

	else
		Copy: s$ + "-shiftformants"
	endif

endproc

procedure extractUV
	runScript: "voicedunvoiced.praat", 0
	select all
	.s_u = selected("Sound", -2)
	.s_v = selected("Sound", -1)
endproc

procedure mixUV
	.sel_tmp = selected("Sound")
	plusObject: extractUV.s_u
	runScript: "copymix.praat", 50, 0, 0
	removeObject: extractUV.s_u, extractUV.s_v, .sel_tmp
endproc
