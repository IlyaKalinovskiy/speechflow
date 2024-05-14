# Parts of this script were adapted from the script "VTchange" by Chris Darwin, https://groups.io/g/Praat-Users-List/files/Darwin%20scripts

form Change vocal tract size
	positive Formant_shift_ratio 1.2
	boolean Preview_(click_Apply._Uncheck_to_publish) 1
endform

formant_shift_ratio = min(formant_shift_ratio, 3)

include batch.praat

procedure action
	s = selected("Sound")
	s$ = selected$("Sound")
	original_dur = Get total duration

include preview1.inc

	if formant_shift_ratio <> 1
		runScript: "workpre.praat"
		wrk = selected("Sound")
		dur1 = Get total duration
		sf = Get sampling frequency

include minmaxf0.praat

		pitch = noprogress To Pitch: 0.01, minF0, maxF0
		f0 = Get quantile: 0, 0, 0.50, "Hertz"

		if f0 <> undefined
			plusObject: wrk
			manipulation = noprogress To Manipulation

			if formant_shift_ratio > 1
				formula$ = "self / formant_shift_ratio"
				rdur = formant_shift_ratio
				rsf = sf / formant_shift_ratio
			elsif formant_shift_ratio < 1
				formula$ = "self * (1 - formant_shift_ratio + 1)"
				rdur = 1 / (1 - formant_shift_ratio + 1)
				rsf = sf * (1 - formant_shift_ratio + 1)
			endif

			pitchtier = Extract pitch tier
			Formula: formula$
			plusObject: manipulation
			Replace pitch tier

			durationtier = Create DurationTier: "tmp", 0, dur1
			Add point: 0, rdur
			plusObject: manipulation
			Replace duration tier

			selectObject: manipulation
			res = Get resynthesis (overlap-add)

			rs = Resample: rsf, 10
			Override sampling frequency: sf

			runScript: "workpost.praat", original_dur
			result = selected("Sound")
			removeObject: wrk, pitch, pitchtier, durationtier, res, manipulation, rs
		else
			selectObject: s
			result = Copy: "tmp"
			removeObject: wrk, pitch
		endif
	else
		result = Copy: "tmp"
	endif

include preview2.inc

	if not preview
		Rename: s$ + "-changevtsize_" + string$(formant_shift_ratio)
	endif
endproc
