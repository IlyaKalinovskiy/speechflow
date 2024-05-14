# Vocal tract length estimation was adapted from the procedure described at: https://www.languagebits.com/phonetics-english/resonant-frequencies-and-vocal-tract-length/
# Vocal tract size change was adapted from the script "VTchange" by Chris Darwin, https://groups.io/g/Praat-Users-List/files/Darwin%20scripts

form Combined processes
	comment Characteristics to be copied from the first to the second selected Sound:
	boolean Vocal_tract_size 0
	optionmenu Pitch 1
		option -- None --
		option Contour
		option Median
		option Variation
		option Median and variation
	optionmenu Time 1
		option -- None --
		option Dynamic time warping (DTW)
		option Duration (stretch)
	boolean EQ_curve 0
	comment Modification parameters (non-default values override the above):
	positive Formant_shift_ratio 1.0 (= no change)
	real New_pitch_median_(Hz) 0.0 (= no change)
	real Pitch_variation_(%) 100 (= no change)
	positive Duration_factor 1.0 (= no change)
	comment General settings:
	boolean Trim_initial_and_final_silences_first 0
	optionmenu DTW_slope_constraint 3
		option no restriction
		option 1/3 < slope < 3
		option 1/2 < slope < 2
		option 2/3 < slope < 3/2
	comment Vocal tract length estimation
	positive Calculate_from_formant 4
	comment Formant determination
	positive Maximum_formant_first_Sound_(Hz) 5500 (= adult female)
	positive Maximum_formant_second_Sound_(Hz) 5500 (= adult female)
	boolean Show_info 1
	boolean Preview_(click_Apply._Uncheck_to_publish) 1
endform

if numberOfSelected("Sound") = 2
	s1 = selected("Sound")
	s1$ = selected$("Sound")
	s2 = selected("Sound", 2)
	s2$ = selected$("Sound", 2)

	result$ = s2$ + "-combinedprocesses-" + s1$

	selectObject: s1
	dur1 = Get total duration
	sf1 = Get sampling frequency
	int1 = Get intensity (dB)

	selectObject: s2
	dur2 = Get total duration
	original_dur = dur2
	sf2 = Get sampling frequency
	int2 = Get intensity (dB)

	plusObject: s1

	formant_shift_ratio = min(formant_shift_ratio, 3)
	new_pitch_median = max(new_pitch_median, 0)
	pitch_variation_factor = pitch_variation / 100

	custom_formant_shift_ratio = 0
	custom_new_pitch_median = 0
	custom_pitch_variation = 0
	custom_duration_factor = 0

	if formant_shift_ratio <> 1
		custom_formant_shift_ratio = 1
		vocal_tract_size = 0
	endif

	if new_pitch_median <> 0
		custom_new_pitch_median = 1
	endif

	if pitch_variation_factor <> 1
		custom_pitch_variation = 1
	endif

	if custom_new_pitch_median or custom_pitch_variation
		if custom_new_pitch_median and custom_pitch_variation
			if pitch <> 2
				pitch = 1
			endif
		else
			if custom_new_pitch_median
				if pitch = 3
					pitch = 1
				endif
				if pitch = 5
					pitch = 4
				endif
			else
				if pitch = 4
					pitch = 1
				endif
				if pitch = 5
					pitch = 3
				endif
			endif
		endif
	endif

	if duration_factor <> 1
		custom_duration_factor = 1
		time = 1
	endif

	if custom_formant_shift_ratio or custom_new_pitch_median or custom_pitch_variation or custom_duration_factor or vocal_tract_size or pitch <> 1 or eQ_curve or time <> 1
		if show_info
			if preview
				appendInfoLine: "-- Combined processes -- (preview)"
			else
				appendInfoLine: "-- Combined processes --"
			endif
		endif

		if int1 = undefined or int2 = undefined
			if show_info
				appendInfoLine: newline$, "Undefined intensity found. Script exited.", newline$, newline$, newline$
			endif

			exitScript()
		endif

		if trim_initial_and_final_silences_first
			if show_info
				appendInfoLine: newline$, "Trimming initial and final silences..."
				appendInfoLine: tab$, s1, ". Sound ", s1$
				appendInfoLine: tab$, tab$, "Total duration: ", dur1, " seconds"
			endif

			selectObject: s1
			s1_original = s1
			s1_trimmed = nocheck nowarn Trim silences: 0.08, "yes", 100, 0, -35, 0.1, 0.05, "no", "trimmed"
			if s1_trimmed = s1 or s1_trimmed = undefined
				s1 = Copy: s1$ + "_trimmed"
			else
				s1 = selected("Sound")
			endif
			s1$ = selected$("Sound")
			s1_dur = Get total duration
			s1_stt = Get start time
			if s1_stt <> 0
				Scale times to: 0, s1_dur
			endif

			if show_info
				appendInfoLine: tab$, tab$, "Trimmed duration: ", s1_dur, " seconds", newline$
				appendInfoLine: tab$, s2, ". Sound ", s2$
				appendInfoLine: tab$, tab$, "Total duration: ", dur2, " seconds"
			endif

			selectObject: s2
			s2_original = s2
			s2_trimmed = nocheck nowarn Trim silences: 0.08, "yes", 100, 0, -35, 0.1, 0.05, "no", "trimmed"
			if s2_trimmed = s2 or s2_trimmed = undefined
				s2 = Copy: s2$ + "_trimmed"
			else
				s2 = selected("Sound")
			endif
			s2$ = selected$("Sound")
			s2_dur = Get total duration
			s2_stt = Get start time
			if s2_stt <> 0
				Scale times to: 0, s2_dur
			endif

			dur1 = s1_dur
			dur2 = s2_dur
			original_dur = dur2

			if show_info
				appendInfoLine: tab$, tab$, "Trimmed duration: ", s2_dur, " seconds"
			endif
		endif

		if pitch <> 1 or time = 2
			selectObject: s1
			runScript: "workpre.praat"
			wrk1 = selected("Sound")
		endif

		wrk1_dur = dur1 + 0.025 + 0.025

		selectObject: s2
		runScript: "workpre.praat"
		wrk2 = selected("Sound")
		wrk2_dur = Get total duration

		if custom_formant_shift_ratio
			if show_info
				appendInfoLine: newline$, "Change vocal tract size..."
				appendInfoLine: tab$, "Formant shift ratio applied: ", formant_shift_ratio
			endif
		else
			if vocal_tract_size
				if show_info
					appendInfoLine: newline$, "Copy vocal tract size..."
					appendInfoLine: tab$, s1, ". Sound ", s1$
				endif

				selectObject: s1
				@getvtl: calculate_from_formant, maximum_formant_first_Sound
				vtl_1 = getvtl.vtl
				vtlr_1 = number(fixed$(17.5 / vtl_1, 2))
				freq_1 = getvtl.formant_frequency

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

				formant_shift_ratio = number(fixed$(vtlr_1 - vtlr_2 + 1, 2))

				if show_info
					appendInfoLine: tab$, tab$, "Estimated vocal tract length: ", number(fixed$(vtl_2, 2)), " cm   (mean F", calculate_from_formant, " = ", number(fixed$(freq_2, 3)), " Hz)"
					appendInfoLine: tab$, tab$, "Vocal tract length ratio: ", vtlr_2, "   (17.5 cm [ref. length] / ", number(fixed$(vtl_2, 2)), " cm)", newline$
					appendInfoLine: tab$, "Formant shift ratio applied: ", formant_shift_ratio, "   (", vtlr_1, " - ", vtlr_2, " + 1)"
				endif
			endif
		endif

		if formant_shift_ratio > 1
			rdur = formant_shift_ratio
		elsif formant_shift_ratio < 1
			rdur = 1 / (1 - formant_shift_ratio + 1)
		elsif formant_shift_ratio = 1
			rdur = 1
		endif

		if custom_duration_factor
			original_dur = dur2 * duration_factor
			new_dur = (dur2 * duration_factor) + 0.025 + 0.025
			dur_factor = new_dur / wrk2_dur
		else
			if time <> 1
				original_dur = dur1
				new_dur = wrk1_dur
				dur_factor = wrk1_dur / wrk2_dur
			else
				new_dur = wrk2_dur
				dur_factor = duration_factor
			endif
		endif

		selectObject: wrk2
		if time = 2
			wrk3 = Copy: "wrk3"
		else
			if rdur * dur_factor > 3
				wrk3 = Extract part: 0, new_dur, "rectangular", 1, "no"
				wrk2_dur = new_dur
			else
				wrk3 = Copy: "wrk3"
			endif
		endif

include minmaxf0.praat

		pitch_2 = noprogress To Pitch: 0.01, minF0, maxF0
		f0_2 = Get quantile: 0, 0, 0.50, "Hertz"
		sd_2 = Get standard deviation: 0, 0, "semitones"

		plusObject: wrk3
		manipulation = noprogress To Manipulation

		if pitch <> 1
			if show_info
				if pitch = 2
					appendInfoLine: newline$, "Copy pitch contour..."
				else
					if pitch = 3
						appendInfoLine: newline$, "Copy pitch median..."
					elsif pitch = 4
						appendInfoLine: newline$, "Copy pitch variation..."
					elsif pitch = 5
						appendInfoLine: newline$, "Copy pitch median and variation..."
					endif
					appendInfoLine: tab$, s1, ". Sound ", s1$
				endif
			endif

			selectObject: wrk1

include minmaxf0.praat

			pitch_1 = noprogress To Pitch: 0.01, minF0, maxF0
			f0_1 = Get quantile: 0, 0, 0.50, "Hertz"
			sd_1 = Get standard deviation: 0, 0, "semitones"

			if f0_1 <> undefined
				if pitch = 2
					pitchtier_1 = Down to PitchTier
				endif

				if show_info
					if pitch <> 2
						if pitch = 3 or pitch = 5
							appendInfoLine: tab$, tab$, "Median pitch: ", number(fixed$(f0_1, 3)), " Hz"
						endif
						if pitch = 4 or pitch = 5
							appendInfoLine: tab$, tab$, "Standard deviation: ", number(fixed$(sd_1, 3)), " semitones"
						endif
						appendInfoLine: newline$, tab$, s2, ". Sound ", s2$
					endif
				endif

				if f0_2 <> undefined
					if show_info
						if pitch = 3 or pitch = 5
							appendInfoLine: tab$, tab$, "Median pitch: ", number(fixed$(f0_2, 3)), " Hz"
						endif
						if pitch = 4 or pitch = 5
							appendInfoLine: tab$, tab$, "Standard deviation: ", number(fixed$(sd_2, 3)), " semitones"
						endif
					endif

					if number(fixed$(f0_1, 3)) <> number(fixed$(f0_2, 3)) or number(fixed$(sd_1, 3)) <> number(fixed$(sd_2, 3))
						if pitch = 3 or pitch = 5
							new_pitch_median = f0_1
						endif
						if pitch = 4 or pitch = 5
							pitch_variation_factor = sd_1 / sd_2
						endif
					endif

					if show_info
						if pitch = 2
							appendInfoLine: tab$, "OK"
						else
							appendInfoLine: ""
							if pitch = 3 or pitch = 5
								appendInfoLine: tab$, "New pitch median: ", number(fixed$(f0_1, 3)), " Hz"
							endif
							if pitch = 4 or pitch = 5
								appendInfoLine: tab$, "Pitch variation applied: ", round((sd_1 / sd_2) * 100), "%   (", number(fixed$(sd_1, 3)), " semitones / ", number(fixed$(sd_2, 3)), " semitones)"
							endif
						endif
					endif
				else
					if trim_initial_and_final_silences_first
						selectObject: s1_original, s2_original
						removeObject: s1, s2
					else
						selectObject: s1, s2
					endif

					if pitch <> 1 or time = 2
						removeObject: wrk1
					endif
					removeObject: wrk2, wrk3, pitch_2, manipulation, pitch_1
					if pitch = 2
						removeObject: pitchtier_1
					endif

					if show_info
						if pitch <> 2
							appendInfo: tab$
						endif
						appendInfoLine: tab$, "There were no voiced segments found. Script exited.", newline$, newline$, newline$
					endif

					exitScript()
				endif
			else
				if trim_initial_and_final_silences_first
					selectObject: s1_original, s2_original
					removeObject: s1, s2
				else
					selectObject: s1, s2
				endif

				if pitch <> 1 or time = 2
					removeObject: wrk1
				endif
				removeObject: wrk2, wrk3, pitch_2, manipulation, pitch_1

				if show_info
					if pitch <> 2
						appendInfo: tab$
					endif
					appendInfoLine: tab$, "There were no voiced segments found. Script exited.", newline$, newline$, newline$
				endif

				exitScript()
			endif
		endif

		if custom_new_pitch_median or custom_pitch_variation
			if show_info
				if custom_new_pitch_median and custom_pitch_variation
					appendInfoLine: newline$, "Change pitch median and variation..."
				else
					if custom_new_pitch_median
						appendInfoLine: newline$, "Change pitch median..."
					else
						appendInfoLine: newline$, "Change pitch variation..."
					endif
				endif
				if custom_new_pitch_median
					appendInfoLine: tab$, "New pitch median: ", new_pitch_median, " Hz"
				endif
				if custom_pitch_variation
					appendInfoLine: tab$, "Pitch variation applied: ", pitch_variation, "%"
				endif
			endif
		endif

		if custom_duration_factor
			if show_info
				appendInfoLine: newline$, "Change duration (stretch)..."
				appendInfoLine: tab$, "New duration: ", dur2 * duration_factor, " seconds   (", dur2, " seconds * ", duration_factor, ")"
				appendInfoLine: tab$, "Duration factor applied: ", duration_factor
			endif
		else
			if time = 2
				if show_info
					appendInfoLine: newline$, "Dynamic time warping (DTW)..."
				endif

				selectObject: wrk1
				Copy: "wrk1_tmp"

				if sf1 <> sf2
					tmp = selected("Sound")
					Resample: sf2, 50
					removeObject: tmp
				endif
				wrk1_tmp = selected("Sound")

				plusObject: wrk2
				time_step = 0.005
				dtw = noprogress To DTW: 0.015, time_step, 0.1, dTW_slope_constraint$

				n_frames = ceiling(wrk1_dur / time_step)
				x_time = 0

				for i to n_frames + 1
					y_time[i] = Get y time from x time: x_time
					x_time += time_step
				endfor

				for i to n_frames
					time_r[i] = time_step / (y_time[i + 1] - y_time[i])
				endfor

				if pitch = 2
					selectObject: pitchtier_1
					n_points = Get number of points
					for i to n_points
						pitchtier_hz[i] = Get value at index: i
						pitchtier_tim[i] = Get time from index: i
					endfor

					selectObject: dtw
					for i to n_points
						pitchtier_y_time[i] = Get y time from x time: pitchtier_tim[i]
					endfor

					selectObject: pitchtier_1
					Remove points between: 0, wrk1_dur
					for i to n_points
						Add point: pitchtier_y_time[i], pitchtier_hz[i]
					endfor
				endif

				removeObject: wrk1_tmp, dtw

				if show_info
					appendInfoLine: tab$, "OK"
				endif
			endif

			if time = 3
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
		endif

		if pitch <> 1 or time = 2
			removeObject: wrk1
		endif
		removeObject: wrk2, wrk3, pitch_2
		if pitch <> 1
			removeObject: pitch_1
		endif

		selectObject: manipulation
		@manipulate

		if eQ_curve
			if show_info
				appendInfoLine: newline$, "Copy EQ curve..."
			endif

			result_tmp = selected("Sound")
			plusObject: s1
			runScript: "copyeq.praat"
			removeObject: result_tmp

			if show_info
				appendInfoLine: tab$, "OK"
			endif
		endif

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
			if trim_initial_and_final_silences_first
				selectObject: s1_original, s2_original
				removeObject: s1, s2
			else
				selectObject: s1, s2
			endif
			removeObject: trimmed, pre, result
		else
			Rename: result$
		endif

		if show_info
			appendInfoLine: newline$, "Completed.", newline$
			if vocal_tract_size
				appendInfoLine: "> Vocal tract length estimation was adapted from the procedure described at:"
				appendInfoLine: "> https://www.languagebits.com/?p=1057", newline$
			endif
			appendInfoLine: newline$
		endif
	endif
endif

procedure getvtl: .fn, .mf
	.sel_tmp = selected("Sound")

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

	.prep = 35000 * ((.fn / 2) - 0.25)
	.vtl = .prep / .formant_frequency
endproc

procedure manipulate
	if pitch = 2
		selectObject: pitchtier_1
	else
		.pitchtier = Extract pitch tier
	endif

	if formant_shift_ratio <> 1
		if formant_shift_ratio > 1
			formula$ = "self / formant_shift_ratio"
		elsif formant_shift_ratio < 1
			formula$ = "self * (1 - formant_shift_ratio + 1)"
		endif
		Formula: formula$
	endif

	if new_pitch_median <> 0
		if pitch = 2
			.f0_f = new_pitch_median / f0_1
		else
			.f0_f = new_pitch_median / f0_2
		endif
		Formula: "self * .f0_f"
	endif

	if pitch_variation_factor <> 1
		if new_pitch_median = 0
			if pitch = 2
				.pm = number(fixed$(f0_1, 3))
			else
				.pm = number(fixed$(f0_2, 3))
			endif
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

	if time = 2
		for .i to n_frames
			if .i = 1
				Add point: y_time[.i], time_r[.i] * rdur
			else
				if number(fixed$(time_r[.i], 6)) <> number(fixed$(time_r[.i - 1], 6))
					Add point: y_time[.i] + 0.00000000001, time_r[.i] * rdur
				endif
			endif
			if .i = n_frames
				Add point: y_time[.i + 1], time_r[.i] * rdur
			else
				if number(fixed$(time_r[.i], 6)) <> number(fixed$(time_r[.i + 1], 6))
					Add point: y_time[.i + 1], time_r[.i] * rdur
				endif
			endif
		endfor
	else
		Add point: 0, rdur * dur_factor
	endif

	plusObject: manipulation
	Replace duration tier

	selectObject: manipulation
	.res = Get resynthesis (overlap-add)

	if formant_shift_ratio <> 1
		if formant_shift_ratio > 1
			rsf = sf2 / formant_shift_ratio
		elsif formant_shift_ratio < 1
			rsf = sf2 * (1 - formant_shift_ratio + 1)
		endif
		.rs = Resample: rsf, 10
		Override sampling frequency: sf2
	endif

	.dur2 = Get total duration
	if .dur2 <> new_dur
		.tmp = Extract part: 0, new_dur, "rectangular", 1, "no"
	endif

	runScript: "workpost.praat", original_dur

	removeObject: manipulation, .durationtier, .res
	if pitch = 2
		removeObject: pitchtier_1
	else
		removeObject: .pitchtier
	endif
	if formant_shift_ratio <> 1
		removeObject: .rs
	endif
	if .dur2 <> new_dur
		removeObject: .tmp
	endif
endproc
