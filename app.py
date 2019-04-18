# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, abort, redirect, Response, url_for
import json
from werkzeug.contrib.cache import SimpleCache

import web
import luyin as ly

cache = SimpleCache()
app = Flask(__name__)

@app.route("/")
def hello():
	return render_template("index.html")

@app.route("/s",methods=['GET', 'POST'])
def s():
	if request.method == "POST":
		filename = request.get_json()['name']
		label = web.predict_with_stack_model(filename)
		return label
	else:
		return redirect("/")



@app.route("/r",methods=['POST'])
def r():
	if request.method == "POST":
		filename = request.get_json()['name']
		ly.record(filename)
		#label = web.predict_with_stack_model("static/"+filename+".wav")
		return "done"
	else:
		return redirect("/")


# @app.route("/player",methods=['GET'])
# def player():
# 	album_id = request.args.get("album")
# 	playlist_id = request.args.get("playlist")
# 	song_id = request.args.get("song")
# 	program_id = request.args.get("program")
# 	radio_id = request.args.get("radio")
# 	mv_id = request.args.get("mv")
#
# 	if album_id is not None:
# 		album_info = netease.netease_cloud_music("album",album_id,0)
# 		songs_info = album_info["songs_info"]
# 		title = "%s - %s" %(album_info["album"],album_info["artist"])
# 		showlrc = "0"
# 	elif playlist_id is not None:
# 		playlist_info = netease.netease_cloud_music("playlist",playlist_id,0)
# 		songs_info = playlist_info["songs_info"]
# 		title = playlist_info["playlist"]
# 		showlrc = "0"
# 	elif song_id is not None:
# 		song_info = netease.netease_cloud_music("song",song_id,1)
# 		title = "%s - %s" %(song_info["title"],song_info["artist"])
# 		songs_info = [song_info]
# 		showlrc = "1"
# 	elif program_id is not None:
# 		song_info = netease.netease_cloud_music("program",program_id,0)
# 		title = song_info["album"]
# 		songs_info = [song_info]
# 		showlrc = "0"
# 	elif radio_id is not None:
# 		songs_info = netease.netease_cloud_music("radio",radio_id,0)
# 		title = songs_info[0]["artist"]
# 		showlrc = "0"
# 	elif mv_id is not None:
# 		mv_info = netease.netease_cloud_music("mv",mv_id,0)
# 		mv_url = mv_info["url_best"]
# 		title = mv_info["title"]
# 		pic_url = mv_info["pic_url"]
# 		return render_template("dplayer.html",mv_url=mv_url,title=title,mv_id=mv_id,pic_url=pic_url)
# 	else:
# 		abort(404)
# 	return render_template("aplayer.html",songs_info=songs_info,title=title,showlrc=showlrc,song_id=song_id)
#
# @app.route("/iframe",methods=['GET'])
# def iframe():
# 	album_id = request.args.get("album")
# 	playlist_id = request.args.get("playlist")
# 	song_id = request.args.get("song")
# 	program_id = request.args.get("program")
# 	radio_id = request.args.get("radio")
# 	mv_id = request.args.get("mv")
#
# 	qssl = request.args.get("qssl")
# 	qlrc = request.args.get("qlrc")
# 	qnarrow = request.args.get("qnarrow")
# 	max_width = request.args.get("max_width")
# 	max_height = request.args.get("max_height")
# 	mode = request.args.get("mode")
# 	autoplay = request.args.get("autoplay")
#
# 	if qnarrow is None:
# 		qnarrow = "false"
# 	else:
# 		pass
# 	if qlrc is None:
# 		qlrc = "0"
# 	else:
# 		pass
# 	if max_width is None:
# 		max_width = "100%"
# 	else:
# 		pass
# 	if mode is None:
# 		mode = "circulation"
# 	else:
# 		pass
# 	if autoplay is None:
# 		autoplay = "true"
# 	else:
# 		pass
# 	if album_id is not None:
# 		album_info = netease.netease_cloud_music("album",album_id,0)
# 		songs_info = album_info["songs_info"]
# 		if qssl == "1":
# 			for i in range(len(songs_info)):
# 				songs_info[i]["url_best"] = songs_info[i]["url_best"].replace('http', 'https')
# 				songs_info[i]["pic_url"] = songs_info[i]["pic_url"].replace('http', 'https')
# 		else:
# 			pass
# 		title = "%s - %s" %(album_info["album"],album_info["artist"])
# 		showlrc = "0"
# 	elif playlist_id is not None:
# 		playlist_info = netease.netease_cloud_music("playlist",playlist_id,0)
# 		songs_info = playlist_info["songs_info"]
# 		if qssl == "1":
# 			for i in range(len(songs_info)):
# 				songs_info[i]["url_best"] = songs_info[i]["url_best"].replace('http', 'https')
# 				songs_info[i]["pic_url"] = songs_info[i]["pic_url"].replace('http', 'https')
# 		else:
# 			pass
# 		title = playlist_info["playlist"]
# 		showlrc = "0"
# 	elif song_id is not None:
# 		song_info = netease.netease_cloud_music("song",song_id,1)
# 		title = "%s - %s" %(song_info["title"],song_info["artist"])
# 		songs_info = [song_info]
# 		if qssl == "1":
# 			songs_info[0]["url_best"] = songs_info[0]["url_best"].replace('http', 'https')
# 			songs_info[0]["pic_url"] = songs_info[0]["pic_url"].replace('http', 'https')
# 		else:
# 			pass
# 		showlrc = qlrc
# 	elif program_id is not None:
# 		song_info = netease.netease_cloud_music("program",program_id,0)
# 		title = song_info["album"]
# 		songs_info = [song_info]
# 		if qssl == "1":
# 			for i in range(len(songs_info)):
# 				songs_info[i]["url_best"] = songs_info[i]["url_best"].replace('http', 'https')
# 				songs_info[i]["pic_url"] = songs_info[i]["pic_url"].replace('http', 'https')
# 		else:
# 			pass
# 		showlrc = "0"
# 	elif radio_id is not None:
# 		songs_info = netease.netease_cloud_music("radio",radio_id,0)
# 		title = songs_info[0]["artist"]
# 		if qssl == "1":
# 			for i in range(len(songs_info)):
# 				songs_info[i]["url_best"] = songs_info[i]["url_best"].replace('http', 'https')
# 				songs_info[i]["pic_url"] = songs_info[i]["pic_url"].replace('http', 'https')
# 		else:
# 			pass
# 		showlrc = "0"
# 	elif mv_id is not None:
# 		mv_info = netease.netease_cloud_music("mv",mv_id,0)
# 		mv_url = mv_info["url_best"]
# 		title = mv_info["title"]
# 		pic_url = mv_info["pic_url"]
# 		return render_template("dplayer_iframe.html",mv_url=mv_url,title=title,mv_id=mv_id,pic_url=pic_url,max_width=max_width,autoplay=autoplay)
# 	else:
# 		abort(404)
#
# 	return render_template("aplayer_iframe.html",songs_info=songs_info,title=title,showlrc=showlrc,qnarrow=qnarrow,max_width=max_width,max_height=max_height,song_id=song_id,autoplay=autoplay,mode=mode)

if __name__ == "__main__":
	app.run()
