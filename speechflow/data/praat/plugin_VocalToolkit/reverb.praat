form Reverb
	comment The following list is updated when Praat starts up with the reverb
	comment impulse response files found in a subfolder called “reverb”.
	optionmenu Preset 1
include reverbpresetslist.inc
	real Mix_(%) 50
	boolean Preview_(click_Apply._Uncheck_to_publish) 1
endform

mix = min(max(mix, 0), 100)

include batch.praat

procedure action
	s = selected("Sound")
	s$ = selected$("Sound")
	int = Get intensity (dB)

	if int <> undefined

include preview1.inc

		wrk = Copy: "wrk"
		runScript: "fixdc.praat"

		reverb_ir = Read from file: "reverb/" + preset$ + ".wav"
		sf1 = 1 / object[wrk].dx
		sf2 = 1 / object[reverb_ir].dx

		selectObject: wrk
		if sf1 <> sf2
			rs = Resample: sf2, 50
		else
			rs = Copy: "tmp"
		endif

		plusObject: reverb_ir
		reverb = Convolve: "sum", "zero"
		Scale intensity: int

		plusObject: rs
		runScript: "copymix.praat", mix, 1, 0
		result = selected("Sound")

		removeObject: wrk, reverb_ir, rs, reverb

include preview2.inc

		if not preview
			Rename: s$ + "-reverb_" + preset$ + "__" + string$(mix)
		endif
	else
		if not preview
			Copy: "tmp"
			Rename: s$ + "-reverb_" + preset$ + "__" + string$(mix)
		endif
	endif
endproc
