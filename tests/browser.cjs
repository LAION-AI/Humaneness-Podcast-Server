// Real public API + real generated MP3. Receipts remain outside the repository.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('fs'),path=require('path');
(async()=>{
const receipt=JSON.parse(fs.readFileSync(process.env.PODCAST_TEST_RECEIPT,'utf8'));
const base=process.env.PODCAST_TEST_URL||'https://arrive-locations-parade-research.trycloudflare.com/podcast/';
const browser=await chromium.launch({executablePath:process.env.CHROME_BIN||'/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-gpu','--mute-audio','--autoplay-policy=no-user-gesture-required']});
const page=await browser.newPage({viewport:{width:1280,height:900}});let errors=[];page.on('pageerror',e=>errors.push(e.message));
await page.goto(base,{waitUntil:'networkidle'});
await page.locator('#validate').click();await page.waitForFunction(()=>document.querySelector('#validation').textContent.includes('Valid:'));
await page.evaluate(j=>localStorage.setItem('podcast-job',JSON.stringify(j)),receipt);await page.reload({waitUntil:'networkidle'});
await page.locator('#result:not([hidden])').waitFor({timeout:60000});
await page.locator('audio').evaluate(a=>a.play());
await page.waitForFunction(()=>document.querySelector('audio').currentTime>1,undefined,{timeout:20000});
const audio=await page.evaluate(()=>{const a=document.querySelector('audio');a.pause();return {time:a.currentTime,duration:a.duration,ready:a.readyState,error:a.error?.message}});
if(!(audio.time>1 && audio.duration>5 && !audio.error))throw Error('Actual MP3 playback failed: '+JSON.stringify(audio));
const download=await Promise.all([page.waitForEvent('download'),page.locator('#report').click()]);
await page.setViewportSize({width:390,height:844});await page.waitForTimeout(300);
const mobile=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth}));
if(mobile.scroll>mobile.width+1||errors.length)throw Error(JSON.stringify({errors,mobile}));
const report={tested_at:new Date().toISOString(),public_url:base,script_validation:true,private_completed_job_recovery_in_browser:true,actual_mp3_playback:audio,report_download:!!download[0],mobile,console_errors:errors};
fs.writeFileSync(process.env.PODCAST_TEST_RESULT||'/tmp/podcast-browser-result.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
