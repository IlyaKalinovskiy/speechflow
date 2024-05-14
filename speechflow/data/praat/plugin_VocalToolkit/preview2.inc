if preview
	pre2 = Extract part: 0, preview_dur, "rectangular", 1, "no"
	nowarn Fade in: 0, 0, 0.025, "yes"
	nowarn Fade out: 0, preview_dur, -0.025, "yes"
	Play
	selectObject: s
	removeObject: trimmed, pre1, result, pre2
endif
