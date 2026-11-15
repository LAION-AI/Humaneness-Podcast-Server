"""Actual HTTPS playback of an authored completed cleanup job; no fake audio.

Install Playwright in a test environment. The capability receipt stays private;
the result includes no job token, script text or source recordings.
"""
import argparse
import asyncio
import json
from pathlib import Path


async def run(args):
    from playwright.async_api import async_playwright
    receipt=json.loads(args.receipt.read_text())
    errors=[]
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=args.chrome,headless=True,
            args=['--no-sandbox','--disable-gpu','--mute-audio','--autoplay-policy=no-user-gesture-required'])
        page=await browser.new_page(viewport={'width':1280,'height':900})
        page.on('pageerror',lambda error:errors.append(str(error)))
        response=await page.goto(args.base,wait_until='networkidle')
        assert response.status==200
        await page.locator('#validate').click()
        await page.wait_for_function('document.querySelector("#validation").textContent.includes("Valid:")')
        await page.evaluate('(receipt)=>localStorage.setItem("podcast-job",JSON.stringify(receipt))',receipt)
        await page.reload(wait_until='networkidle')
        await page.locator('#result:not([hidden])').wait_for(timeout=60000)
        await page.locator('audio').evaluate('(a)=>a.play()')
        await page.wait_for_function('document.querySelector("audio").currentTime>1',timeout=20000)
        audio=await page.locator('audio').evaluate('(a)=>{a.pause();return {time:a.currentTime,duration:a.duration,ready:a.readyState,error:!!a.error}}')
        assert audio['time']>1 and audio['duration']>5 and not audio['error']
        async with page.expect_download() as download:
            await page.locator('#report').click()
        report=await download.value
        assert report.suggested_filename.endswith('.json')
        await page.set_viewport_size({'width':390,'height':844})
        overflow=await page.evaluate('document.documentElement.scrollWidth>innerWidth+1')
        assert not overflow and not errors
        result=dict(passed=True,public_url=args.base,actual_mp3_playback=audio,
                    report_download=True,mobile_overflow=overflow,page_errors=errors,
                    physical_audio_device_test=False)
        args.result.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result))
        await browser.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--base',default='https://arrive-locations-parade-research.trycloudflare.com/podcast/')
    parser.add_argument('--chrome',default='/usr/bin/google-chrome')
    asyncio.run(run(parser.parse_args()))
