- map.osm: openstreetmapからOverPass APIを使って抽出したネットワークデータ
- osm_map.html: map.osmを読み込んでhtmlで表示したもの
- output.net.xml: map.osmからSUMOシミュレータで使えるネットワーク形式に変換したもの
- output.geojson: SUMOに付属の、net2geojson.pyツールを用いてoutput.net.xmlをgeojson形式に変換したもの。交差点の処理が怪しいので、シミュレーションの結果とネットワークデータを紐づける時以外は使わない

