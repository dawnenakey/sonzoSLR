# Bug Fixes Applied

## ✅ Fixed Issues

### 1. Logo Manifest Error (FIXED)
**Problem:** SVG icon was causing manifest validation error in PWA
**Fix:** Removed SVG from manifest.json, using PNG icons instead
**Status:** ✅ Committed and pushed to GitHub
**Commit:** `ae9a149`

### 2. CORS Policy Error (PARTIALLY FIXED)
**Problem:** API Gateway blocking video requests from Amplify app
**Fix:** Ran `scripts/fix-cors.sh` to add CORS headers to `/videos` endpoint
**Status:** ⚠️ Script partially succeeded, but deployment may be needed
**Action Required:** 
- Check if video fetching works now
- If still failing, manually deploy API Gateway changes in AWS Console

## 🔧 Remaining Issues to Fix

Based on the feedback in your Google Sheet:

### 3. Camera Permission Denied (MacBook Pro M1 Pro)
**Problem:** Webcam not accessible despite being available
**Location:** `/CameraTest` and `/camera` pages
**Possible Causes:**
- Browser permissions not granted
- HTTPS required for camera access
- macOS security settings blocking access

**Fix Attempt:**
```bash
# Check if running on HTTPS (required for camera)
# Amplify should be using HTTPS automatically
```

### 4. Video Upload/Playback Issues
**Problem:** Can't play/load uploaded videos
**Location:** `/home` page
**Details:**
- Works for MP4
- Doesn't work for MOV
- Error: "No video ID was provided in the URL"

**Needs Investigation:**
- Check video format handling in upload logic
- Verify video ID is being passed correctly in ambient
- Check if MOV files need transcoding

### 5. Missing Video ID Error
**Problem:** "Error: No video ID was provided in the URL"
**Location:** Annotation page after upload
**Possible Cause:** URL parameters not being preserved when navigating to annotator

**Potential Fix:** Ensure video ID is passed in URL when navigating to annotator

## 🎯 Next Steps

1. **Test CORS Fix:**
   - Refresh your Amplify app
   - Check if video fetching works now
   - If not, go to AWS Console and deploy the API Gateway stage

2. **Camera Issues:**
   - Verify you're accessing the app via HTTPS (not HTTP)
   - Grant camera permissions in browser settings
   - Check macOS Security & Privacy settings for camera access

3. **Video Format Support:**
   - Investigate MOV vs MP4 handling
   - Consider adding format validation/conversion

4. **URL Parameter Handling:**
   - Fix navigation to pass video ID correctly
   - Add error handling for missing video ID

## 📊 Impact Summary

- **Critical Bugs:** 5 identified
- **Fixed:** 1 (logo manifest)
- **Partially Fixed:** 1 (CORS - needs verification)
- **Remaining:** 3 (camera, video format, URL params)

## 🔍 How to Verify Fixes

1. **Logo:** Check browser tab icon and app headers
2. **CORS:** Try fetching videos from the home page
3. **Camera:** Try accessing webcam from CameraTest page
4. **Video Upload:** Upload an MP4 and verify playback
5. **Annotation:** Upload a video and verify navigation to annotator

