from mido import MidiFile, MidiTrack, Message
import numpy as np
import torch

N_NOTES = 96
N_MEASURES = 96

def midi_to_samples(fname):
    has_time_sig = False
    flag_warning = False
    try:
        mid = MidiFile(fname)
    except:
        print(f"Corrupt file, skipping {fname}")
        return []
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
        print("  ^^^^^^ WARNING ^^^^^^")
        print("    " + fname)
        print("    Detected multiple distinct time signatures.")
        print("  ^^^^^^ WARNING ^^^^^^")
        return []
    
    all_notes = {}
    for track in mid.tracks:
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            if msg.type == 'note_on':
                if msg.velocity == 0:
                    continue
                note = int(msg.note - (128 - N_NOTES)/2)
                if not (note >= 0 and note < N_NOTES):
                    return []
                if note not in all_notes:
                    all_notes[note] = []
                else:
                    single_note = all_notes[note][-1]
                    if len(single_note) == 1:
                        single_note.append(single_note[0] + 1)
                all_notes[note].append([abs_time * N_MEASURES / ticks_per_measure])
            elif msg.type == 'note_off':
                try:
                    if len(all_notes[note][-1]) != 1:
                        continue
                    all_notes[note][-1].append(abs_time * N_MEASURES / ticks_per_measure)
                except:
                    continue
    for note in all_notes:
        for start_end in all_notes[note]:
            if len(start_end) == 1:
                start_end.append(start_end[0] + 1)

    samples = []
    for note in all_notes:
        for start, _ in all_notes[note]:
            sample_ix = int(start / N_MEASURES)
            while len(samples) <= sample_ix:
                samples.append(np.zeros((N_MEASURES, N_NOTES), dtype=np.uint8))
            sample = samples[sample_ix]
            start_ix = int(start - sample_ix * N_MEASURES)
            sample[start_ix, note] = 1
    return samples


def samples_to_midi(samples, fname, thresh=0.5):
    samples = samples.numpy()
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    ticks_per_sample = (mid.ticks_per_beat * 4) / N_MEASURES
    abs_time = 0
    last_time = 0
    last_tick = samples[0].shape[0] - 1

    for j, sample in enumerate(samples):
        s = np.argwhere(sample>thresh)
        for i in range(len(s)):
            y, x = s[i]
            abs_time = y * ticks_per_sample + (j*N_NOTES*ticks_per_sample)
            note = int(x + (128 - N_NOTES)/2)
            if y == 0 or sample[y-1,x] < thresh:
                delta_time = int(abs_time - last_time)
                track.append(Message('note_on', note=note, velocity=127, time=delta_time))
                last_time = abs_time
            if y == last_tick or sample[y+1, x] < thresh:
                delta_time = int(abs_time - last_time)
                track.append(Message('note_off', note=note, velocity=127, time=delta_time))
                last_time = abs_time

    mid.save(fname)
    print("Midi saved to:", fname)

