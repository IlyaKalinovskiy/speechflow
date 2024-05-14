form Vibrato and tremolo
	comment Vibrato (pitch variation)
	real Semitones_(0-12) 1
	comment Tremolo (intensity variation)
	real Decibels_(0-12) 1
	comment Vibrato and tremolo rate
	real Pulses_per_second_(1-10) 5.5
	boolean Preview_(click_Apply._Uncheck_to_publish) 1
endform

semitones = min(max(semitones, 0), 12)
decibels = min(max(decibels, 0), 12)
pulses_per_second = min(max(pulses_per_second, 1), 10)
pulses = pi * (pulses_per_second * 2) / 100

include batch.praat

procedure action
	s = selected("Sound")
	s$ = selected$("Sound")
	original_dur = Get total duration
	int = Get intensity (dB)

include preview1.inc

	runScript: "workpre.praat"
	wrk = selected("Sound")
	dur = Get total duration

include minmaxf0.praat

	pitch = noprogress To Pitch: 0.01, minF0, maxF0
	f0 = Get quantile: 0, 0, 0.50, "Hertz"

	if f0 <> undefined
		plusObject: wrk
		manipulation = noprogress To Manipulation

		pitchtier = Extract pitch tier
		for i from 0 to dur * 100
			val[i] = Get value at time: i / 100
		endfor

		vibrato = Create PitchTier: "vibrato", 0, dur
		tremolo = Create IntensityTier: "tremolo", 0, dur
		@vibratoTremolo: round(dur * 100), semitones, decibels, pulses

		selectObject: manipulation, vibrato
		Replace pitch tier

		durationtier = Create DurationTier: "tmp", 0, dur
		Add point: 0, 1
		plusObject: manipulation
		Replace duration tier

		selectObject: manipulation
		res = Get resynthesis (overlap-add)

		plusObject: tremolo
		tmp = Multiply: "yes"

		runScript: "workpost.praat", original_dur
		Scale intensity: int
		runScript: "declip.praat"
		result = selected("Sound")

		removeObject: wrk, pitch, manipulation, pitchtier, vibrato, tremolo, durationtier, res, tmp
	else
		removeObject: wrk, pitch
		selectObject: s
		result = Copy: "tmp"
	endif

include preview2.inc

	if not preview
		Rename: s$ + "-vibratotremolo"
	endif
endproc

procedure vibratoTremolo: .tim, .vib, .db, .pul
	selectObject: vibrato
	.ramp = 0
	for .i from 0 to .tim - 1
		if .vib <> 0
			if .i <= .tim - 25 and .ramp <= 1
				.ramp = .ramp + 0.04
				if .ramp > 1
					.ramp = 1
				endif
			else
				.ramp = .ramp - 0.04
				if .ramp < 0
					.ramp = 0
				endif
			endif
			.b = 12 * ln(val[.i] / 261.63) / ln(2)
			.c = .b + (.vib / 2) * sin(.pul * .i) * .ramp
			.ptch = 261.63 * exp(.c * ln(2) / 12)
		else
			.ptch = val[.i]
		endif
		Add point: .i / 100, .ptch
	endfor

	if .db <> 0
		selectObject: tremolo
		.ramp = 0
		for .i from 0 to .tim - 1
			if .i <= .tim - 25 and .ramp <= 1
				.ramp = .ramp + 0.04
				if .ramp > 1
					.ramp = 1
				endif
			else
				.ramp = .ramp - 0.04
				if .ramp < 0
					.ramp = 0
				endif
			endif
			.int = 90 + (.db / 2) * sin(.pul * .i) * .ramp
			Add point: .i / 100, .int
		endfor
	endif
endproc
