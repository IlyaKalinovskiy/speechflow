# Vocal tract length estimation was adapted from the procedure described at: https://www.languagebits.com/phonetics-english/resonant-frequencies-and-vocal-tract-length/

form Copy vocal tract size
	comment Vocal tract length estimation
	positive Calculate_from_formant 4
	comment Formant determination
	positive Maximum_formant_first_Sound_(Hz) 5500 (= adult female)
	positive Maximum_formant_second_Sound_(Hz) 5500 (= adult female)
	comment Set 5000 Hz for men, 5500 Hz for women or up to 8000 Hz for children.
	boolean Show_info 1
	boolean Preview_(click_Apply._Uncheck_to_publish) 1
endform

if numberOfSelected("Sound") = 2
	s1 = selected("Sound")
	s1$ = selected$("Sound")
	s2 = selected("Sound", 2)
	s2$ = selected$("Sound", 2)

	if show_info
		if preview
			appendInfoLine: "Copy vocal tract size... (preview)"
		else
			appendInfoLine: "Copy vocal tract size..."
		endif
		appendInfoLine: tab$, s1, ". Sound ", s1$
	endif

	selectObject: s1
	@getvtl: calculate_from_formant, maximum_formant_first_Sound
	vtl_1 = getvtl.vtl
	vtlr_1 = number(fixed$(17.5 / vtl_1, 2))
	freq_1 = getvtl.formant_frequency

	if freq_1 <> undefined
		if show_info
			appendInfoLine: tab$, tab$, "Estimated vocal tract length: ", number(fixed$(vtl_1, 2)), " cm   (mean F", calculate_from_formant, " = ", number(fixed$(freq_1, 3)), " Hz)"
			appendInfoLine: tab$, tab$, "Vocal tract length ratio: ", vtlr_1, "   (17.5 cm [ref. length] / ", number(fixed$(vtl_1, 2)), " cm)", newline$
			appendInfoLine: tab$, s2, ". Sound ", s2$
		endif

		selectObject: s2
		@getvtl: calculate_from_formant, maximum_formant_second_Sound
		vtl_2 = getvtl.vtl
		vtlr_2 = number(fixed$(17.5 / vtl_2, 2))
		freq_2 = getvtl.formant_frequency

		if freq_2 <> undefined
			if show_info
				appendInfoLine: tab$, tab$, "Estimated vocal tract length: ", number(fixed$(vtl_2, 2)), " cm   (mean F", calculate_from_formant, " = ", number(fixed$(freq_2, 3)), " Hz)"
				appendInfoLine: tab$, tab$, "Vocal tract length ratio: ", vtlr_2, "   (17.5 cm [ref. length] / ", number(fixed$(vtl_2, 2)), " cm)", newline$
			endif

			formant_shift_ratio = number(fixed$(vtlr_1 - vtlr_2 + 1, 2))

			runScript: "changevt.praat", formant_shift_ratio, 0
			result = selected("Sound")

			if show_info
				appendInfoLine: tab$, "Formant shift ratio applied: ", formant_shift_ratio, "   (", vtlr_1, " - ", vtlr_2, " + 1)", newline$
			endif

			if preview
				if show_info
					Rename: "preview"
					appendInfo: tab$, "Playing preview... "
				endif
include preview.inc
				if show_info
					appendInfoLine: "OK", newline$
				endif
				selectObject: s1, s2
				removeObject: trimmed, pre, result
			else
				Rename: s2$ + "-copyvtsize-" + s1$
			endif

			if show_info
				appendInfoLine: "> Vocal tract length estimation was adapted from the procedure described at:"
				appendInfoLine: "> https://www.languagebits.com/?p=1057", newline$, newline$, newline$
			endif
		else
			selectObject: s1, s2

			if show_info
				appendInfoLine: tab$, tab$, "There were no voiced segments found. Script exited.", newline$, newline$, newline$
			endif
		endif
	else
		selectObject: s1, s2

		if show_info
			appendInfoLine: tab$, tab$, "There were no voiced segments found. Script exited.", newline$, newline$, newline$
		endif
	endif
endif

procedure getvtl: .fn, .mf
	.sel_tmp = selected("Sound")
	.int = Get intensity (dB)

	if .int <> undefined
		runScript: "workpre.praat"
		.tmp1 = selected("Sound")

		runScript: "extractvowels.praat", 0, 0
		.tmp2 = selected("Sound")

		runScript: "workpre.praat"
		.tmp3 = selected("Sound")

		.formant = noprogress nowarn To Formant (robust): 0.005, 5, .mf, 0.025, 50, 1.5, 5, 0.000001
		.formant_frequency = Get mean: .fn, 0, 0, "hertz"

		selectObject: .sel_tmp
		removeObject: .tmp1, .tmp2, .tmp3, .formant
	else
		.formant_frequency = undefined
	endif

	.prep = 35000 * ((.fn / 2) - 0.25)
	.vtl = .prep / .formant_frequency
endproc
