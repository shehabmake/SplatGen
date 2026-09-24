"""Build SplatGen's outline-free 24-unit vector icon family.

Run with a development Python containing Pillow; Blender only reads the PNGs.
Every symbol has one source definition below. The generator writes editable
SVGs, transparent 192px PNGs, and a contact sheet showing actual 16/24/32px use.
The existing logo and Windows ICO are hashed for preservation, never rewritten.
"""
from __future__ import annotations

import math
import colorsys
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

# Load the pure palette module without importing Blender's add-on package.
import runpy
_palette = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'palette.py'))
ORANGE, BLUE, LIME, RED, GREEN = (_palette[k] for k in ('ORANGE', 'BLUE', 'YELLOW', 'RED', 'GREEN'))
BRAND_BLUE = BLUE
NEUTRAL = _palette['NEUTRAL']
MUTED = NEUTRAL
PALETTE_RGB = tuple(tuple(round(c * 255) for c in _palette['rgb'](value))
                    for value in _palette['ICON_COLORS'])
# Color marks primary operations and meaningful status, never routine tools.
ACCENT_ICONS = frozenset({
    'section_cameras', 'section_coverage', 'section_dataset',
    'action_close', 'action_delete', 'action_hide', 'action_remove',
    'action_build', 'action_train', 'action_calculate', 'action_coverage',
    'action_resume', 'action_export', 'action_pause', 'action_stop',
    'action_save', 'status_ok', 'status_wait', 'status_error',
    'badge_new', 'coverage_fair', 'coverage_weak',
})
INK = '#161A20'
PAPER = '#F7F8FB'
SIZE = 192
SUPERSAMPLE = 4


class Symbol:
    """Simple vector primitives rendered identically into SVG and PNG."""
    def __init__(self):
        self.items = []

    def line(self, points, color=BLUE, width=2.2, closed=False):
        self.items.append(('line', tuple(points), color, width, closed))
        return self

    def circle(self, x, y, r, color=BLUE, fill=False, width=2.2):
        self.items.append(('circle', (x, y, r), color, width, fill))
        return self

    def polygon(self, points, color=ORANGE):
        self.items.append(('polygon', tuple(points), color, 0, True))
        return self

    def arc(self, x, y, r, start, end, color=BLUE, width=2.2):
        steps = max(4, int(abs(end-start) / 4))
        return self.line([(x+r*math.cos(math.radians(start+(end-start)*i/steps)),
                           y+r*math.sin(math.radians(start+(end-start)*i/steps)))
                         for i in range(steps+1)], color, width)

    def box(self, x0, y0, x1, y1, color=BLUE, width=2.2):
        return self.line([(x0,y0),(x1,y0),(x1,y1),(x0,y1)],color,width,True)

    def png(self, size=SIZE):
        # Rasterize coverage independently from colour. Only alpha changes at
        # the edge; RGB remains the shape's true hue, never a white/black matte.
        # Area reduction avoids sharpening-ring halos in the authored artwork.
        scale = size*SUPERSAMPLE/24
        image = Image.new('RGBA', (size,size))
        xy = lambda p: [(round(x*scale),round(y*scale)) for x,y in p]
        for kind,points,color,width,closed in self.items:
            mask = Image.new('L',(size*SUPERSAMPLE,)*2)
            draw = ImageDraw.Draw(mask)
            if kind == 'line':
                pts = points + (points[0],) if closed else points
                draw.line(xy(pts),fill=255,width=max(1,round(width*scale)),joint='curve')
                r = width*scale/2
                for x,y in pts:
                    draw.ellipse((x*scale-r,y*scale-r,x*scale+r,y*scale+r),fill=255)
            elif kind == 'circle':
                x,y,r=points
                if closed:
                    rr=r*scale
                    draw.ellipse((x*scale-rr,y*scale-rr,x*scale+rr,y*scale+rr),fill=255)
                else:
                    rr=(r+width/2)*scale
                    draw.ellipse((x*scale-rr,y*scale-rr,x*scale+rr,y*scale+rr),outline=255,width=max(1,round(width*scale)))
            else:
                draw.polygon(xy(points),fill=255)
            layer = Image.new('RGBA',(size,size),color)
            layer.putalpha(mask.resize((size,size),Image.Resampling.BOX))
            image=Image.alpha_composite(image,layer)
        pixels = list(image.getdata())
        chosen = {}
        def exact(pixel):
            if not pixel[3]:
                return (0, 0, 0, 0)
            rgb = pixel[:3]
            if rgb not in chosen:
                chosen[rgb] = min(PALETTE_RGB, key=lambda color: sum((a-b)**2 for a,b in zip(rgb,color)))
            return (*chosen[rgb], pixel[3])
        image.putdata([exact(pixel) for pixel in pixels])
        return image

    def svg(self):
        parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke-linecap="round" stroke-linejoin="round">']
        for kind,points,color,width,closed in self.items:
            if kind == 'line':
                pts=points+(points[0],) if closed else points
                value=' '.join(f'{x:.3f},{y:.3f}' for x,y in pts)
                parts.append(f'<polyline points="{value}" stroke="{color}" stroke-width="{width}"/>')
            elif kind == 'circle':
                x,y,r=points
                if closed:
                    parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>')
                else:
                    parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" stroke="{color}" stroke-width="{width}"/>')
            else:
                value=' '.join(f'{x:.3f},{y:.3f}' for x,y in points)
                parts.append(f'<polygon points="{value}" fill="{color}"/>')
        return '\n'.join(parts+['</svg>'])+'\n'


def camera(color=BLUE):
    return (Symbol().line([(3,7),(8,7),(10,4),(15,4),(17,7),(21,7),(21,19),(3,19)],color,2,True)
            .circle(12,12.5,3.4,color,False,1.7))


def layers(color=ORANGE):
    return (Symbol().line([(3,8),(12,3),(21,8),(12,13)],color,2,True)
            .line([(3,12),(12,17),(21,12)],color,2)
            .line([(3,16),(12,21),(21,16)],color,2))


def play(color=GREEN):
    return Symbol().polygon([(6,3.5),(21,12),(6,20.5)],color)


def export(up=True):
    s=Symbol().line([(4,15),(4,20),(20,20),(20,15)],ORANGE,2)
    if up:
        s.line([(12,15),(12,3)],ORANGE,2.5).line([(7,8),(12,3),(17,8)],ORANGE,2.5)
    else:
        s.line([(12,3),(12,15)],BLUE,2.5).line([(7,10),(12,15),(17,10)],BLUE,2.5)
    return s


def recolor(symbol, color):
    result = Symbol()
    result.items = [(kind, points, color, width, closed)
                    for kind, points, _old, width, closed in symbol.items]
    return result


def symbols():
    icons={}
    icons['stage_project']=Symbol().line([(3,8),(3,5),(10,5),(12,8),(21,8),(19,20),(3,20),(3,8),(21,8)],BLUE,2)
    icons['stage_cameras']=camera()
    icons['stage_dataset']=layers()
    icons['stage_points']=Symbol()
    for x,y,r,c in ((5,7,1.8,BLUE),(13,4,1.6,BLUE),(20,9,1.8,ORANGE),(11,13,2.1,ORANGE),(4,19,1.6,BLUE),(18,20,1.8,BLUE)):
        icons['stage_points'].circle(x,y,r,c,True)
    icons['stage_training']=Symbol().arc(12,12,8.5,25,155,GREEN,2.2).arc(12,12,8.5,205,335,GREEN,2.2).polygon([(9,7),(17,12),(9,17)],GREEN)
    eye=[(2+20*i/20,12-6*math.sin(math.pi*i/20)) for i in range(21)]
    eye += [(2+20*i/20,12+6*math.sin(math.pi*i/20)) for i in range(19,-1,-1)]
    icons['stage_viewer']=Symbol().line(eye,BLUE,2.2,True).circle(12,12,2.6,BLUE,True)
    icons['stage_export']=export()
    icons['camera_pending']=camera(ORANGE)
    icons['camera_rendered']=camera(BLUE).line([(15,17),(18,20),(23,14)],BLUE,2.2)
    icons['action_build']=layers(GREEN)
    icons['action_train']=play()
    icons['action_import']=export(False)
    icons['action_export']=recolor(export(), GREEN)
    icons['action_save']=recolor(export(), LIME)
    icons['action_pause']=Symbol().line([(8,5),(8,19)],LIME,3.5).line([(16,5),(16,19)],LIME,3.5)
    icons['action_stop']=Symbol().polygon([(5,5),(19,5),(19,19),(5,19)],RED)
    icons['action_resume']=play(GREEN)
    icons['action_add']=Symbol().line([(12,4),(12,20)],GREEN,2.5).line([(4,12),(20,12)],GREEN,2.5)
    icons['action_remove']=Symbol().line([(5,12),(19,12)],RED,2.5)
    icons['action_frame']=Symbol().line([(8,3),(3,3),(3,8)],BLUE,2.3).line([(16,3),(21,3),(21,8)],BLUE,2.3).line([(3,16),(3,21),(8,21)],BLUE,2.3).line([(16,21),(21,21),(21,16)],BLUE,2.3)
    icons['ui_settings']=Symbol()
    for x,y in ((5,8),(12,16),(19,10)):
        icons['ui_settings'].line([(x,3),(x,21)],BLUE,1.8).circle(x,y,2.5,BLUE,True)
    icons['ui_help']=Symbol().circle(12,12,9,BLUE,False,2).line([(9,8),(10.5,6.8),(13.5,6.8),(15,8.3),(15,10),(12,12.5),(12,13.5)],BLUE,2).circle(12,17.3,1,BLUE,True)
    icons['ui_search']=Symbol().circle(10,10,6,BLUE,False,2).line([(14.5,14.5),(21,21)],BLUE,2.5)
    icons['ui_changed']=Symbol().polygon([(3,4),(21,4),(14,13),(14,19),(10,21),(10,13)],LIME)
    icons['ui_reset']=Symbol().arc(12,12,8,-140,180,BLUE,2).line([(4,3),(4,9),(10,9)],BLUE,2)
    icons['ui_save']=Symbol().line([(4,3),(17,3),(21,7),(21,21),(3,21),(3,3),(4,3)],BLUE,2).box(8,3,16,9,BLUE,1.6).box(7,14,17,21,BLUE,1.6)
    icons['ui_save']=recolor(icons['ui_save'], LIME)
    icons['status_active']=Symbol().arc(12,12,8,-65,220,GREEN,3)
    icons['status_ok']=Symbol().line([(4,12),(9.5,18),(21,5)],GREEN,3.1)
    icons['status_wait']=Symbol().circle(12,12,8,LIME,False,2).line([(12,7),(12,12),(16,14)],LIME,2)
    icons['status_error']=Symbol().line([(12,3),(22,21),(2,21)],RED,2.1,True).line([(12,9),(12,14)],RED,2.3).circle(12,18,.9,RED,True)
    icons['status_idle']=Symbol().circle(12,12,6,MUTED,False,2)
    icons['badge_new']=Symbol().polygon([(12,2),(14.6,8.5),(22,12),(14.6,15.5),(12,22),(9.4,15.5),(2,12),(9.4,8.5)],LIME)
    def ellipse(cx,cy):
        angle=math.radians(-30)
        return [(cx+8*math.cos(t)*math.cos(angle)-4.6*math.sin(t)*math.sin(angle),
                 cy+8*math.cos(t)*math.sin(angle)+4.6*math.sin(t)*math.cos(angle))
                for t in (2*math.pi*i/64 for i in range(64))]
    icons['layer_splats']=Symbol().polygon(ellipse(12,8),ORANGE).polygon(ellipse(12,16),BLUE)
    icons['layer_coverage']=Symbol().circle(12,12,9,BLUE,False,2).line([(7,12),(10.5,15.5),(17,8.5)],GREEN,2.4)
    icons['layer_scene']=Symbol().line([(12,2),(21,7),(21,17),(12,22),(3,17),(3,7),(12,2),(12,12),(21,7),(12,12),(3,7),(12,12),(12,22)],BLUE,1.8)
    icons['viewer_relight']=(Symbol().arc(12,9.5,4.6,150,390,ORANGE,2)
        .line([(8,11.8),(9.2,16),(14.8,16),(16,11.8)],ORANGE,2)
        .line([(9.5,19),(14.5,19)],ORANGE,2)
        .line([(11,21.5),(13,21.5)],ORANGE,2)
        .line([(12,1),(12,2)],ORANGE,1.8)
        .line([(2,8),(4,8)],ORANGE,1.8).line([(20,8),(22,8)],ORANGE,1.8)
        .line([(4.5,2.5),(6,4)],ORANGE,1.8).line([(18,4),(19.5,2.5)],ORANGE,1.8))
    # Auto camera rig: viewpoints placed around a subject - an orbit with
    # three camera positions (build orange) around the scanned scene (blue).
    icons['method_auto_rig']=Symbol().arc(12,12,8.2,-60,15,ORANGE,2).arc(12,12,8.2,45,135,ORANGE,2).arc(12,12,8.2,165,240,ORANGE,2).circle(12,12,2.8,BLUE,True)
    for angle in (-90,30,150):
        icons['method_auto_rig'].circle(12+8.2*math.cos(math.radians(angle)),12+8.2*math.sin(math.radians(angle)),2.3,ORANGE,True)
    # Scan blob: Blender's sphere Empty - an outline, its equator, a centre.
    icons['method_blob']=Symbol().circle(12,12,8.5,BLUE,False,2).line([(12+8.5*math.cos(2*math.pi*i/48),12+3.1*math.sin(2*math.pi*i/48)) for i in range(48)],BLUE,1.5,True).circle(12,12,2.1,BLUE,True)
    # Bug body, wing seam, antennae and three pairs of legs.
    icons['ui_bug']=(Symbol().circle(12,6,2.5,BLUE,False,1.8)
        .line([(8,9),(16,9),(17,13),(16,18),(12,21),(8,18),(7,13),(8,9)],BLUE,1.8)
        .line([(12,10),(12,20)],BLUE,1.6)
        .line([(10,4),(8,2)],BLUE,1.8).line([(14,4),(16,2)],BLUE,1.8))
    for y in (10,14,18):
        icons['ui_bug'].line([(4,y-1),(8,y)],BLUE,1.8).line([(16,y),(20,y-1)],BLUE,1.8)
    # A launch arrow, clearly different from the path field's folder picker.
    icons['action_open_folder']=(Symbol().line([(20,14),(19,20),(3,20),(3,5),(9,5),(12,8)],MUTED,2)
        .line([(12,14),(21,5),(15,5),(21,5),(21,11)],BLUE,2.2))
    # Color variants retain the original geometry and UI layout.
    icons['action_calculate']=recolor(icons['method_auto_rig'], GREEN)
    icons['action_coverage']=recolor(icons['layer_coverage'], GREEN)
    icons['coverage_fair']=recolor(icons['status_idle'], LIME)
    icons['coverage_weak']=recolor(icons['status_wait'], ORANGE)
    icons['section_cameras']=recolor(icons['stage_cameras'], ORANGE)
    icons['section_coverage']=recolor(icons['layer_coverage'], LIME)
    icons['section_dataset']=recolor(icons['stage_dataset'], GREEN)
    icons['action_close']=Symbol().line([(5,5),(19,19)],RED,2.4).line([(19,5),(5,19)],RED,2.4)
    icons['action_delete']=(Symbol().line([(5,7),(6,21),(18,21),(19,7)],RED,2)
        .line([(3,5),(21,5)],RED,2).line([(9,5),(9,2),(15,2),(15,5)],RED,1.8)
        .line([(10,10),(10,17)],RED,1.6).line([(14,10),(14,17)],RED,1.6))
    icons['action_hide']=recolor(icons['stage_viewer'], RED).line([(3,21),(21,3)],RED,2.4)
    return {name: symbol if name in ACCENT_ICONS else recolor(symbol, NEUTRAL)
            for name, symbol in icons.items()}


def contact_sheet(target, icons):
    width,row_h=1020,46
    sheet=Image.new('RGB',(width,105+len(icons)*row_h),'#25272A')
    draw=ImageDraw.Draw(sheet)
    try:
        font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',14)
        title=ImageFont.truetype('C:/Windows/Fonts/seguisb.ttf',23)
    except OSError:
        font=title=ImageFont.load_default()
    draw.text((18,15),'SplatGen 5.3 · pipeline accents and clear actions',font=title,fill=PAPER)
    draw.text((18,48),'Delivered PNGs at 16 / 24 / 32 px. No light outlines, dark outlines, tile backgrounds or glow.',font=font,fill='#B7C1CC')
    for index,(background,label,ink) in enumerate((('#25272A','Dark',PAPER),('#D8DADE','Light',INK),('#30363E','Selected',PAPER))):
        left=index*340
        draw.rectangle((left,78,left+339,sheet.height),fill=background)
        draw.text((left+15,82),label,font=font,fill=ink)
    for n,(name,symbol) in enumerate(icons.items()):
        y=105+n*row_h
        source=Image.open(target/f'{name}.png')
        for col in range(3):
            draw.text((col*340+15,y+14),name,font=font,fill=INK if col==1 else PAPER)
            for index,size in enumerate((16,24,32)):
                icon=source.resize((size,size),Image.Resampling.LANCZOS)
                sheet.paste(icon,(col*340+202+index*40,y+(row_h-size)//2),icon)
    sheet.save(target/'icon_review.png')


def edge_review(target):
    """Actual-size icons beside enlarged pixel coverage, with no faux outline."""
    sheet=Image.new('RGB',(1000,470),'#25272A')
    draw=ImageDraw.Draw(sheet)
    try:
        font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',16)
    except OSError:
        font=ImageFont.load_default()
    for col,name in enumerate(('viewer_relight','status_error','ui_search')):
        x=col*250
        draw.text((x+12,10),name,font=font,fill=PAPER)
        source=Image.open(target/f'{name}.png').convert('RGBA')
        for n,size in enumerate((16,24,32)):
            native=source.resize((size,size),Image.Resampling.LANCZOS)
            sheet.paste(native,(x+30+n*60,47),native)
        native=source.resize((24,24),Image.Resampling.LANCZOS)
        large=native.resize((192,192),Image.Resampling.NEAREST)
        # Checkerboard is composited underneath; it is not in the PNG asset.
        for iy in range(12):
            for ix in range(12):
                color='#3A3D43' if (ix+iy)%2 else '#25272A'
                draw.rectangle((x+24+ix*16,104+iy*16,x+39+ix*16,119+iy*16),fill=color)
        sheet.paste(large,(x+24,104),large)
        draw.text((x+24,305),'24px, nearest ×8',font=font,fill=PAPER)
        alpha=native.getchannel('A').resize((96,96),Image.Resampling.NEAREST).convert('RGB')
        sheet.paste(alpha,(x+24,343))
        draw.text((x+127,363),'Alpha',font=font,fill=PAPER)
    sheet.save(target/'icon_edge_review.png')


def main():
    target=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent
    target.mkdir(parents=True,exist_ok=True)
    svg_dir=target/'svg'
    svg_dir.mkdir(exist_ok=True)
    icons=symbols()
    # Guard branding when rebuilding the separate UI symbol family.
    identity={name:hashlib.sha256((target/name).read_bytes()).hexdigest()
              for name in ('splatgen_logo.png','splatgen.ico') if (target/name).exists()}
    for name,symbol in icons.items():
        symbol.png().save(target/f'{name}.png',optimize=True)
        (svg_dir/f'{name}.svg').write_text(symbol.svg(),encoding='utf-8')
    contact_sheet(target,icons)
    edge_review(target)
    report={'palette':{},'single_colour_edges':{},'identity_sha256':identity,'exact_palette_icons':{}}
    for name in icons:
        visible={p[:3] for p in Image.open(target/f'{name}.png').convert('RGBA').getdata() if p[3]}
        assert visible <= set(PALETTE_RGB), (name, visible - set(PALETTE_RGB))
        if name not in ACCENT_ICONS:
            assert visible == {tuple(int(NEUTRAL[i:i+2],16) for i in (1,3,5))}, name
        report['exact_palette_icons'][name] = ['#'+''.join(f'{c:02X}' for c in color) for color in sorted(visible)]
    for name,value in [('orange',ORANGE),('blue',BLUE),('lime',LIME),('green',GREEN),('red',RED)]:
        rgb=tuple(int(value[i:i+2],16) for i in (1,3,5))
        report['palette'][name]={'hex':value,'saturation':round(colorsys.rgb_to_hsv(*(v/255 for v in rgb))[1],4)}
    report['neutral_icons'] = sorted(set(icons) - ACCENT_ICONS)
    report['accent_icons'] = sorted(ACCENT_ICONS)
    report['neutral_color'] = NEUTRAL
    for name,colour in [('stage_cameras',NEUTRAL),('action_train',GREEN),('status_error',RED),('viewer_relight',NEUTRAL)]:
        pixels=list(Image.open(target/f'{name}.png').convert('RGBA').getdata())
        expected=tuple(int(colour[i:i+2],16) for i in (1,3,5))
        edges=[pixel for pixel in pixels if 0<pixel[3]<255]
        assert edges and all(pixel[:3]==expected for pixel in pixels if pixel[3]),name+' has a colour matte or pale edge'
        report['single_colour_edges'][name]={'partial_alpha_pixels':len(edges),'all_visible_rgb_matches_fill':True}
    assert identity=={name:hashlib.sha256((target/name).read_bytes()).hexdigest() for name in identity}
    (target/'ui_palette_review.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f'Generated {len(icons)} SVG/PNG pairs and actual-size/alpha review sheets. Logo and ICO untouched.')


if __name__=='__main__':
    main()
