"""Video backup from real browser captures. Scenes, URLs and timestamps stay auditable."""
from pathlib import Path
import argparse
import json
import subprocess


def srt_time(seconds):
    return f'{seconds//3600:02}:{seconds//60%60:02}:{seconds%60:02},000'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('scenes',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();scenes=json.loads(a.scenes.read_text(encoding='utf-8'))
    if not scenes or any(not s.get('url') or not s.get('captured_utc') for s in scenes):
        raise ValueError('Every frame requires a source URL and capture timestamp')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    from PIL import Image
    normalized=a.scenes.parent/'normalized';normalized.mkdir(exist_ok=True)
    lines=[];captions=[];elapsed=0
    for i,scene in enumerate(scenes,1):
        source=(a.scenes.parent/scene['file']).resolve();source.relative_to(a.scenes.parent.resolve())
        if not source.is_file():raise FileNotFoundError(source)
        duration=int(scene.get('seconds',30))
        if not 1<=duration<=120:raise ValueError('Invalid scene duration')
        # Browser captures may contain JPEG bytes even with a .png filename.
        # Decode each input independently and preserve its pixels as PNG before
        # concat, which otherwise assumes every input uses the first codec.
        frame=normalized/f'{i:02}.png'
        with Image.open(source) as img:
            img.convert('RGB').save(frame,'PNG')
        file_name=frame.resolve().as_posix().replace("'","'\\''")
        lines.extend(["file '"+file_name+"'",f'duration {duration}'])
        captions.append(f'{i}\n{srt_time(elapsed)} --> {srt_time(elapsed+duration)}\n{scene["caption"]}\n')
        elapsed+=duration
    lines.append("file '"+file_name+"'")
    concat=a.scenes.parent/'frames.ffconcat';concat.write_text('\n'.join(lines),encoding='utf-8')
    subtitles=a.output.with_suffix('.srt');subtitles.write_text('\n'.join(captions),encoding='utf-8')
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-xerror','-y','-f','concat','-safe','0',
       '-i',str(concat),'-i',str(subtitles),'-t',str(elapsed),
       '-vf','scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black',
       '-r','15','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p',
       '-c:s','mov_text','-metadata:s:s:0','language=spa','-disposition:s:0','default','-movflags','+faststart',str(a.output)],check=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-xerror','-i',str(a.output),'-f','null','-'],check=True)
    a.output.with_suffix('.provenance.json').write_text(json.dumps({'kind':'Guided backup from actual browser captures',
       'duration_seconds':elapsed,'scenes':scenes,'audio':False,'claims':'Screens and subtitles from accepted Databricks execution'},indent=2),encoding='utf-8')
    print(json.dumps({'video':str(a.output),'seconds':elapsed,'scenes':len(scenes)}))


if __name__=='__main__':main()
