import numpy as np
from mido import MidiFile, MidiTrack, Message

N_NOTES = 96
SAMPLES_PER_MEASURE = 96

def midi_to_sample(fname):
	has_time_sig = False
	flag_warning = False
	mid = MidiFile(fname)
	ticks_per_beat = mid.ticks_per_beat
	ticks_per_measure = 4 * ticks_per_beat

	for track in mid.tracks:
		for msg in track:
			if msg.type == 'time_signature':
				new_tpm = msg.numerator * ticks_per_beat * 4 / msg.denominator
				if has_time_sig and new_tpm != ticks_per_measure:
					flag_warning = True
				ticks_per_measure = new_tpm
				has_time_sig = True

	if flag_warning:
		print(f"ERROR detected: multiple time signatures in {fname}, returning empty list.")
		return []


	all_notes = dict()
	for track in mid.tracks:
		abs_time = 0
		for msg in track:
			abs_time += msg.time
			if msg.type == "note_on":
				if msg.velocity == 0:
					continue
				note = int(msg.note - (128 - N_NOTES) / 2)
				assert(note >= 0 and note < N_NOTES)
				if note not in all_notes:
					all_notes[note] = []
				else:
					single_note = all_notes[note][-1]
					if len(single_note) == 1:
						single_note.append(single_note[0] + 1)
				all_notes[note].append([abs_time * SAMPLES_PER_MEASURE / ticks_per_measure])
			elif msg.type == "note_off":
				if len(all_notes[note]) == 1:
					continue
				all_notes[note][-1].append(abs_time * SAMPLES_PER_MEASURE / ticks_per_measure)

	for note in all_notes.keys():
		for start_end in all_notes[note]:
			if len(start_end) == 1:
				start_end.append(start_end[0] + 1)
			elif len(start_end) == 3:
				start_end.pop(1)

	samples = []
	for note in all_notes.keys():
		for start, end in all_notes[note]:
			sample_ix = int(start / SAMPLES_PER_MEASURE)
			while len(samples) <= sample_ix:
				samples.append(np.zeros((SAMPLES_PER_MEASURE, N_NOTES), dtype=np.uint8))

			sample = samples[sample_ix]
			start_ix = int(start - sample_ix * SAMPLES_PER_MEASURE)
			end_ix = int(min(end - sample_ix * SAMPLES_PER_MEASURE, SAMPLES_PER_MEASURE))
			for i in range(start_ix, end_ix):
				sample[i, note] = 1

	return samples


def samples_to_midi(samples, fname, thresh=0.5):
	mid = MidiFile()
	track = MidiTrack()
	mid.tracks.append(track)
	ticks_per_beat = mid.ticks_per_beat
	ticks_per_measure = 4 * ticks_per_beat
	ticks_per_sample = int(ticks_per_measure / SAMPLES_PER_MEASURE)
	abs_time = 0
	last_time = 0
	for sample in samples:
		for y in range(sample.shape[0]):
			abs_time += ticks_per_sample
			for x in range(sample.shape[1]):
				note = int(x + (128 - N_NOTES)/2)
				if sample[y,x] >= thresh and (y == 0 or sample[y-1,x] < thresh):
					delta_time = abs_time - last_time
					track.append(Message('note_on', note=note, velocity=127, time=delta_time))
					last_time = abs_time
				if sample[y,x] >= thresh and (y == sample.shape[0]-1 or sample[y+1,x] < thresh):
					delta_time = abs_time - last_time
					track.append(Message('note_off', note=note, velocity=127, time=delta_time))
					last_time = abs_time
	mid.save(fname)
