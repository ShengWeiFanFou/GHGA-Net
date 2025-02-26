import os
import struct
import filetype

''' support type

Image
dwg - image/vnd.dwg
xcf - image/x-xcf
jpg - image/jpeg
jpx - image/jpx
png - image/png
apng - image/apng
gif - image/gif
webp - image/webp
cr2 - image/x-canon-cr2
tif - image/tiff
bmp - image/bmp
jxr - image/vnd.ms-photo
psd - image/vnd.adobe.photoshop
ico - image/x-icon
heic - image/heic
avif - image/avif

Video
3gp - video/3gpp
mp4 - video/mp4
m4v - video/x-m4v
mkv - video/x-matroska
webm - video/webm
mov - video/quicktime
avi - video/x-msvideo
wmv - video/x-ms-wmv
mpg - video/mpeg
flv - video/x-flv

Audio
aac - audio/aac
mid - audio/midi
mp3 - audio/mpeg
m4a - audio/mp4
ogg - audio/ogg
flac - audio/x-flac
wav - audio/x-wav
amr - audio/amr
aiff - audio/x-aiff

Archive
br - application/x-brotli
rpm - application/x-rpm
dcm - application/dicom
epub - application/epub+zip
zip - application/zip
tar - application/x-tar
rar - application/x-rar-compressed
gz - application/gzip
bz2 - application/x-bzip2
7z - application/x-7z-compressed
xz - application/x-xz
pdf - application/pdf
exe - application/x-msdownload
swf - application/x-shockwave-flash
rtf - application/rtf
eot - application/octet-stream
ps - application/postscript
sqlite - application/x-sqlite3
nes - application/x-nintendo-nes-rom
crx - application/x-google-chrome-extension
cab - application/vnd.ms-cab-compressed
deb - application/x-deb
ar - application/x-unix-archive
Z - application/x-compress
lzo - application/x-lzop
lz - application/x-lzip
lz4 - application/x-lz4
zstd - application/zstd

Document
doc - application/msword
docx - application/vnd.openxmlformats-officedocument.wordprocessingml.document
odt - application/vnd.oasis.opendocument.text
xls - application/vnd.ms-excel
xlsx - application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
ods - application/vnd.oasis.opendocument.spreadsheet
ppt - application/vnd.ms-powerpoint
pptx - application/vnd.openxmlformats-officedocument.presentationml.presentation
odp - application/vnd.oasis.opendocument.presentation

Font
woff - application/font-woff
woff2 - application/font-woff
ttf - application/font-sfnt
otf - application/font-sfnt

Application
wasm - application/wasm  '''


# 这个库通过文件头判断 可以解决篡改文件后缀名的情况
def get_file_type(file_path):
    type=filetype.guess(file_path)
    if type!=None:
        return type

#
# type=get_file_type("../uploads/dc.txt")
# type=get_file_type(r"D:\bishesystem\毕设\topsecret.txt")
# type=get_file_type('../文本.png')
# print('文件类型为：'+type.extension)
# print('文件属于：'+type.mime)