form Vocoder
	positive Frequency_(Hz) 130.81
	optionmenu Carrier_waveform 1
		option Pulse train
		option Sawtooth
		option Square
		option Triangle
		option Hum
		option Phonation
		option White noise
	boolean Preview_(click_Apply._Uncheck_to_publish) 1
endform

include batch.praat

procedure action
	s = selected("Sound")
	s$ = selected$("Sound")
	int = Get intensity (dB)

	if int <> undefined

include preview1.inc

		dur = Get total duration
		sf = Get sampling frequency

		runScript: "createwaveform.praat", dur, sf, frequency, 0.2, 0, 0, carrier_waveform$, 0
		carrier = selected("Sound")
		Scale intensity: int

		if preview
			selectObject: pre1
		else
			selectObject: s
		endif
		tmp = Copy: "tmp"
		plusObject: carrier

		runScript: "copyvocoder.praat"
		result = selected("Sound")

		removeObject: carrier, tmp

include preview2.inc

		if not preview
			Rename: s$ + "-vocoder_" + carrier_waveform$
		endif
	else
		if not preview
			Copy: s$ + "-vocoder_" + carrier_waveform$
		endif
	endif
endproc
