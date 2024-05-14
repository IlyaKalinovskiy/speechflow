form Copy pitch median and variation
	boolean Copy_pitch_median 1
	boolean Copy_pitch_variation 1
	boolean Show_info 1
	boolean Preview_(click_Apply._Uncheck_to_publish) 1
endform

if numberOfSelected("Sound") = 2
	s1 = selected("Sound")
	s1$ = selected$("Sound")
	s2 = selected("Sound", 2)
	s2$ = selected$("Sound", 2)

	if copy_pitch_median or copy_pitch_variation
		new_pitch_median = 0
		pitch_variation_factor = 1

		if show_info
			if copy_pitch_median and copy_pitch_variation
				if preview
					appendInfoLine: "Copy pitch median and variation... (preview)"
				else
					appendInfoLine: "Copy pitch median and variation..."
				endif
			elsif copy_pitch_median
				if preview
					appendInfoLine: "Copy pitch median... (preview)"
				else
					appendInfoLine: "Copy pitch median..."
				endif
			elsif copy_pitch_variation
				if preview
					appendInfoLine: "Copy pitch variation... (preview)"
				else
					appendInfoLine: "Copy pitch variation..."
				endif
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

			selectObject: s2
			original_dur = Get total duration
			runScript: "workpre.praat"
			wrk2 = selected("Sound")
			dur = Get total duration

include minmaxf0.praat

			pitch_2 = noprogress To Pitch: 0.01, minF0, maxF0
			f0_2 = Get quantile: 0, 0, 0.50, "Hertz"
			sd_2 = Get standard deviation: 0, 0, "semitones"

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
					selectObject: pitch_2
					plusObject: wrk2
					manipulation = noprogress To Manipulation

					if copy_pitch_median
						new_pitch_median = f0_1
					endif
					if copy_pitch_variation
						pitch_variation_factor = sd_1 / sd_2
					endif

					@manipulate
					result = selected("Sound")
				else
					selectObject: s2
					result = Copy: "tmp"
				endif
				removeObject: wrk1, pitch_1, wrk2, pitch_2

				if show_info
					appendInfoLine: ""
					if copy_pitch_median
						appendInfoLine: tab$, "New pitch median: ", number(fixed$(f0_1, 3)), " Hz"
					endif
					if copy_pitch_variation
						appendInfoLine: tab$, "Pitch variation applied: ", round((sd_1 / sd_2) * 100), "%   (", number(fixed$(sd_1, 3)), " semitones / ", number(fixed$(sd_2, 3)), " semitones)"
					endif
				endif

				if preview
					if show_info
						Rename: "preview"
						appendInfo: newline$, tab$, "Playing preview... "
					endif
include preview.inc
					if show_info
						appendInfoLine: "OK"
					endif
					selectObject: s1, s2
					removeObject: trimmed, pre, result
				else
					Rename: s2$ + "-copypitchmedian-" + s1$
				endif

				if show_info
					appendInfoLine: newline$, newline$
				endif
			else
				selectObject: s1, s2
				removeObject: wrk1, pitch_1, wrk2, pitch_2

				if show_info
					appendInfoLine: tab$, tab$, "There were no voiced segments found. Script exited.", newline$, newline$, newline$
				endif
			endif
		else
			selectObject: s1, s2
			removeObject: wrk1, pitch_1

			if show_info
				appendInfoLine: tab$, tab$, "There were no voiced segments found. Script exited.", newline$, newline$, newline$
			endif
		endif
	endif
endif

procedure manipulate
	.pitchtier = Extract pitch tier

	.durationtier = Create DurationTier: "tmp", 0, dur
	Add point: 0, 1
	plusObject: manipulation
	Replace duration tier

	selectObject: .pitchtier

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

		.fref_st = 12 * ln(.pm / 100) / ln(2)
		Formula: "if self <> undefined then 100 * exp((.fref_st + 12 * ln(self / .pm) / ln(2) * pitch_variation_factor) * ln(2) / 12) else self fi"
	endif

	plusObject: manipulation
	Replace pitch tier

	selectObject: manipulation
	.res = Get resynthesis (overlap-add)
	runScript: "workpost.praat", original_dur

	removeObject: manipulation, .pitchtier, .durationtier, .res
endproc
