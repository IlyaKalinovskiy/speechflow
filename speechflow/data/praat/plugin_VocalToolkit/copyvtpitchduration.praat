# Vocal tract length estimation was adapted from the procedure described at: https://www.languagebits.com/phonetics-english/resonant-frequencies-and-vocal-tract-length/

form Copy vocal tract size, pitch and duration
	boolean Copy_vocal_tract_size 1
	boolean Copy_pitch_median 1
	boolean Copy_pitch_variation 1
	boolean Copy_duration_(stretch) 1
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

	if copy_vocal_tract_size or copy_pitch_median or copy_pitch_variation or copy_duration
		if show_info
			if preview
				appendInfoLine: "-- Copy vocal tract size, pitch and duration -- (preview)"
			else
				appendInfoLine: "-- Copy vocal tract size, pitch and duration --"
			endif
		endif

		formant_shift_ratio = 1
		new_pitch_median = 0
		pitch_variation_factor = 1
		new_duration = 0

		if copy_duration
			selectObject: s1
			dur1 = Get total duration

			selectObject: s2
			dur2 = Get total duration

			if dur1 <> dur2
				new_duration = dur1
			endif
		endif

		if copy_vocal_tract_size
			if show_info
				appendInfoLine: newline$, "Copy vocal tract size..."
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
					formant_shift_ratio = number(fixed$(vtlr_1 - vtlr_2 + 1, 2))

					if show_info
						appendInfoLine: tab$, tab$, "Estimated vocal tract length: ", number(fixed$(vtl_2, 2)), " cm   (mean F", calculate_from_formant, " = ", number(fixed$(freq_2, 3)), " Hz)"
						appendInfoLine: tab$, tab$, "Vocal tract length ratio: ", vtlr_2, "   (17.5 cm [ref. length] / ", number(fixed$(vtl_2, 2)), " cm)", newline$
						appendInfoLine: tab$, "Formant shift ratio applied: ", formant_shift_ratio, "   (", vtlr_1, " - ", vtlr_2, " + 1)"
					endif
				else
					selectObject: s1, s2

					if show_info
						appendInfoLine: tab$, tab$, "There were no voiced segments found. Script exited.", newline$, newline$, newline$
					endif

					exitScript()
				endif
			else
				selectObject: s1, s2

				if show_info
					appendInfoLine: tab$, tab$, "There were no voiced segments found. Script exited.", newline$, newline$, newline$
				endif

				exitScript()
			endif
		endif

		selectObject: s2
		original_dur = Get total duration
		runScript: "workpre.praat"
		wrk2 = selected("Sound")
		wrk2_dur = Get total duration
		sf = Get sampling frequency

		if new_duration = 0
			new_dur = wrk2_dur
		else
			new_dur = new_duration + 0.025 + 0.025
			original_dur = new_duration
		endif
		duration_factor = new_dur / wrk2_dur

		if formant_shift_ratio > 1
			rdur = formant_shift_ratio
		elsif formant_shift_ratio < 1
			rdur = 1 / (1 - formant_shift_ratio + 1)
		elsif formant_shift_ratio = 1
			rdur = 1
		endif

		if rdur * duration_factor > 3
			wrk3 = Extract part: 0, new_dur, "rectangular", 1, "no"
			wrk2_dur = new_dur
		else
			wrk3 = Copy: "wrk3"
		endif

include minmaxf0.praat

		pitch_2 = noprogress To Pitch: 0.01, minF0, maxF0
		f0_2 = Get quantile: 0, 0, 0.50, "Hertz"
		sd_2 = Get standard deviation: 0, 0, "semitones"

		plusObject: wrk3
		manipulation = noprogress To Manipulation

		if formant_shift_ratio > 1
			formula$ = "self / formant_shift_ratio"
			rsf = sf / formant_shift_ratio
		elsif formant_shift_ratio < 1
			formula$ = "self * (1 - formant_shift_ratio + 1)"
			rsf = sf * (1 - formant_shift_ratio + 1)
		endif

		removeObject: wrk2, wrk3, pitch_2

		if copy_pitch_median or copy_pitch_variation
			if show_info
				if copy_pitch_median and copy_pitch_variation
					appendInfoLine: newline$, "Copy pitch median and variation..."
				elsif copy_pitch_median
					appendInfoLine: newline$, "Copy pitch median..."
				elsif copy_pitch_variation
					appendInfoLine: newline$, "Copy pitch variation..."
				endif
				appendInfoLine: tab$, s1, ". Sound ", s1$
			endif

			selectObject: s1
			runScript: "workpre.praat"
			wrk1 = selected("Sound")

include minmaxf0.praat

			pitch_1 = noprogress To Pitch: 0.01, minF0, maxF0
			f0_1 = Get quantile: 0, 0, 0.50, "Hertz"
			sd_1 = Get standard deviation: 0, 0, "semitones"

			if f0_1 <> undefined
				if show_info
					if copy_pitch_median
						appendInfoLine: tab$, tab$, "Median pitch: ", number(fixed$(f0_1, 3)), " Hz"
					endif
					if copy_pitch_variation
						appendInfoLine: tab$, tab$, "Standard deviation: ", number(fixed$(sd_1, 3)), " semitones"
					endif
					appendInfoLine: newline$, tab$, s2, ". Sound ", s2$
				endif

				if f0_2 <> undefined
					if show_info
						if copy_pitch_median
							appendInfoLine: tab$, tab$, "Median pitch: ", number(fixed$(f0_2, 3)), " Hz"
						endif
						if copy_pitch_variation
							appendInfoLine: tab$, tab$, "Standard deviation: ", number(fixed$(sd_2, 3)), " semitones"
						endif
					endif

					if number(fixed$(f0_1, 3)) <> number(fixed$(f0_2, 3)) or number(fixed$(sd_1, 3)) <> number(fixed$(sd_2, 3))
						if copy_pitch_median
							new_pitch_median = f0_1
						endif
						if copy_pitch_variation
							pitch_variation_factor = sd_1 / sd_2
						endif
					endif

					if show_info
						appendInfoLine: ""
						if copy_pitch_median
							appendInfoLine: tab$, "New pitch median: ", number(fixed$(f0_1, 3)), " Hz"
						endif
						if copy_pitch_variation
							appendInfoLine: tab$, "Pitch variation applied: ", round((sd_1 / sd_2) * 100), "%   (", number(fixed$(sd_1, 3)), " semitones / ", number(fixed$(sd_2, 3)), " semitones)"
						endif
					endif

					removeObject: wrk1, pitch_1
				else
					selectObject: s1, s2
					removeObject: wrk1, pitch_1, manipulation

					if show_info
						appendInfoLine: tab$, tab$, "There were no voiced segments found. Script exited.", newline$, newline$, newline$
					endif

					exitScript()
				endif
			else
				selectObject: s1, s2
				removeObject: wrk1, pitch_1, manipulation

				if show_info
					appendInfoLine: tab$, tab$, "There were no voiced segments found. Script exited.", newline$, newline$, newline$
				endif

				exitScript()
			endif
		endif

		if copy_duration
			if show_info
				appendInfoLine: newline$, "Copy duration (stretch)..."
				appendInfoLine: tab$, s1, ". Sound ", s1$
				appendInfoLine: tab$, tab$, "Total duration: ", dur1, " seconds", newline$
				appendInfoLine: tab$, s2, ". Sound ", s2$
				appendInfoLine: tab$, tab$, "Total duration: ", dur2, " seconds", newline$
				appendInfoLine: tab$, "New duration: ", dur1, " seconds"
				appendInfoLine: tab$, "Duration factor applied: ", dur1 / dur2, "   (", dur1, " seconds / ", dur2, " seconds)"
			endif
		endif

		selectObject: manipulation
		@manipulate
		result = selected("Sound")

		if preview
			if show_info
				Rename: "preview"
				appendInfo: newline$, "Playing preview... "
			endif
include preview.inc
			if show_info
				appendInfoLine: "OK"
			endif
			selectObject: s1, s2
			removeObject: trimmed, pre, result
		else
			Rename: s2$ + "-copyvtsizepitchduration-" + s1$
		endif

		if show_info
			appendInfoLine: newline$, "Completed.", newline$
			if copy_vocal_tract_size
				appendInfoLine: "> Vocal tract length estimation was adapted from the procedure described at:"
				appendInfoLine: "> https://www.languagebits.com/?p=1057", newline$
			endif
			appendInfoLine: newline$
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

procedure manipulate
	.pitchtier = Extract pitch tier

	if formant_shift_ratio <> 1
		Formula: formula$
	endif

	if new_pitch_median <> 0
		.f0_f = new_pitch_median / f0_2
		Formula: "self * .f0_f"
	endif

	if pitch_variation_factor <> 1
		if new_pitch_median = 0
			.pm = number(fixed$(f0_2, 3))
		else
			.pm = number(fixed$(new_pitch_median, 3))
		endif

		if formant_shift_ratio > 1
			.pm = .pm / formant_shift_ratio
		elsif formant_shift_ratio < 1
			.pm = .pm * (1 - formant_shift_ratio + 1)
		endif

		.fref_st = 12 * ln(.pm / 100) / ln(2)
		Formula: "if self <> undefined then 100 * exp((.fref_st + 12 * ln(self / .pm) / ln(2) * pitch_variation_factor) * ln(2) / 12) else self fi"
	endif

	plusObject: manipulation
	Replace pitch tier

	.durationtier = Create DurationTier: "tmp", 0, wrk2_dur
	Add point: 0, rdur * duration_factor
	plusObject: manipulation
	Replace duration tier

	selectObject: manipulation
	.res = Get resynthesis (overlap-add)

	if formant_shift_ratio <> 1
		.rs = Resample: rsf, 10
		Override sampling frequency: sf
	endif

	.dur2 = Get total duration
	if .dur2 <> new_dur
		.tmp = Extract part: 0, new_dur, "rectangular", 1, "no"
	endif

	runScript: "workpost.praat", original_dur

	removeObject: manipulation, .pitchtier, .durationtier, .res
	if formant_shift_ratio <> 1
		removeObject: .rs
	endif
	if .dur2 <> new_dur
		removeObject: .tmp
	endif
endproc
