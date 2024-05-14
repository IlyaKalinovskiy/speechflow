# This script uses the process to add jitter suggested by Paul Boersma at https://groups.io/g/Praat-Users-List/message/2878

form Raspiness
	real Raspiness_(%) 20
	boolean Preview_(click_Apply._Uncheck_to_publish) 1
endform

raspiness = min(max(raspiness, 0), 100)

include batch.praat

procedure action
	s = selected("Sound")
	s$ = selected$("Sound")
	original_dur = Get total duration

include preview1.inc

	if raspiness <> 0
		runScript: "workpre.praat"
		wrk = selected("Sound")
		dur = Get total duration

include minmaxf0.praat

		pitch = noprogress To Pitch: 0.01, minF0, maxF0
		f0 = Get quantile: 0, 0, 0.50, "Hertz"

		if f0 <> undefined
			plusObject: wrk
			manipulation = noprogress To Manipulation

			durationtier = Create DurationTier: "tmp", 0, dur
			Add point: 0, 1
			plusObject: manipulation
			Replace duration tier

			selectObject: pitch
			pointprocess = noprogress To PointProcess

			matrix = noprogress To Matrix
			r = raspiness / 100000
			Formula: "self + randomGauss(0, r)"

			pointprocess2 = noprogress To PointProcess
			plusObject: manipulation
			Replace pulses

			selectObject: manipulation
			res = Get resynthesis (overlap-add)

			runScript: "workpost.praat", original_dur
			result = selected("Sound")
			removeObject: wrk, pitch, manipulation, durationtier, pointprocess, matrix, pointprocess2, res
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
		Rename: s$ + "-raspiness_" + string$(raspiness)
	endif
endproc
